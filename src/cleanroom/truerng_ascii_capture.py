#!/usr/bin/python3
"""Capture raw ADC samples from TrueRNG Pro in RAW ASCII mode.

This script reads raw 10-bit ADC samples from the TrueRNG Pro device,
parsing the MODE_RAW_ASC format (line-based CSV).
Samples from each generator are saved to separate files.

RAW ASCII Format:
  Each line contains: gen1_value,gen2_value
  Values are 10-bit ADC readings (0-1023)

Usage:
  python truerng_ascii_capture.py [options]

Options:
  --samples N      Number of samples to capture (default: 1000000)
  --normalize      Output normalized floats (0.0-1.0) instead of integers
  --prefix PREFIX  Output file prefix (default: "ascii_adc")
  --port PORT      Serial port (auto-detected if not specified)
"""

import argparse
import sys
import time
from pathlib import Path

import serial

from truerng_utils import (
    get_first_truerng,
    mode_change,
    reset_serial_port,
)

# ADC resolution
ADC_MAX_VALUE = 1023  # 10-bit ADC: 2^10 - 1

# Capture configuration
BLOCK_SIZE = 64 * 1024  # Read 64KB at a time
FLUSH_INTERVAL = 10000  # Flush output every N samples


def parse_ascii_line(line: str) -> tuple[int, int] | None:
    """Parse a single ASCII line into Gen1 and Gen2 samples.

    Args:
        line: A line from MODE_RAW_ASC output (format: "gen1,gen2").

    Returns:
        Tuple of (gen1, gen2) values, or None if parsing fails.
    """
    try:
        parts = line.strip().split(",")
        if len(parts) != 2:
            return None
        gen1 = int(parts[0])
        gen2 = int(parts[1])
        # Validate 10-bit range
        if not (0 <= gen1 <= ADC_MAX_VALUE) or not (0 <= gen2 <= ADC_MAX_VALUE):
            return None
        return gen1, gen2
    except ValueError:
        return None


def parse_ascii_data(
    data: str,
    partial_line: str = "",
) -> tuple[list[int], list[int], str, int]:
    """Parse ASCII data into Gen1 and Gen2 samples.

    Args:
        data: String data from MODE_RAW_ASC.
        partial_line: Incomplete line from previous block.

    Returns:
        Tuple of (gen1_samples, gen2_samples, remaining_partial, invalid_count).
    """
    gen1_samples: list[int] = []
    gen2_samples: list[int] = []
    invalid_count = 0

    # Prepend partial line from previous block
    full_data = partial_line + data

    # Split into lines, keeping track of incomplete final line
    lines = full_data.split("\n")

    # Last element may be incomplete (no newline at end)
    remaining = lines[-1] if not full_data.endswith("\n") else ""
    complete_lines = lines[:-1] if remaining else lines

    for line in complete_lines:
        if not line:
            continue
        result = parse_ascii_line(line)
        if result is not None:
            gen1, gen2 = result
            gen1_samples.append(gen1)
            gen2_samples.append(gen2)
        else:
            invalid_count += 1

    return gen1_samples, gen2_samples, remaining, invalid_count


def format_sample(value: int, normalize: bool) -> str:
    """Format a sample value for output.

    Args:
        value: Raw 10-bit ADC value (0-1023).
        normalize: If True, output as float (0.0-1.0).

    Returns:
        Formatted string representation.
    """
    if normalize:
        return f"{value / ADC_MAX_VALUE:.10f}"
    return str(value)


def capture_ascii_samples(
    port: str,
    num_samples: int,
    output_prefix: str,
    normalize: bool,
) -> tuple[int, int]:
    """Capture ASCII mode ADC samples and save to files.

    Args:
        port: Serial port path.
        num_samples: Target number of samples to capture.
        output_prefix: Prefix for output filenames.
        normalize: If True, output normalized floats.

    Returns:
        Tuple of (gen1_count, gen2_count) samples captured.
    """
    # Switch to RAW ASCII mode
    if not mode_change("MODE_RAW_ASC", port, verbose=True):
        print("Failed to switch to MODE_RAW_ASC")
        return 0, 0

    # Generate output filenames
    timestamp = time.strftime("%Y%m%d.%H%M%S")
    gen1_path = Path(f"{output_prefix}_gen1_{timestamp}.txt")
    gen2_path = Path(f"{output_prefix}_gen2_{timestamp}.txt")

    print(f"Output files: {gen1_path}, {gen2_path}")
    print(f"Target samples: {num_samples:,}")
    print(f"Format: {'normalized (0.0-1.0)' if normalize else 'raw integers (0-1023)'}")

    # Open serial port
    try:
        ser = serial.Serial(port=port, timeout=10)
    except serial.SerialException as e:
        print(f"Failed to open port {port}: {e}")
        return 0, 0

    gen1_count = 0
    gen2_count = 0
    total_invalid = 0
    partial_line = ""

    try:
        if not ser.isOpen():
            ser.open()

        ser.setDTR(True)
        ser.flushInput()

        start_time = time.time()

        with open(gen1_path, "w") as f1, open(gen2_path, "w") as f2:
            while gen1_count < num_samples:
                # Read a block of data
                try:
                    raw_data = ser.read(BLOCK_SIZE)
                except serial.SerialException as e:
                    print(f"\nRead error: {e}")
                    break

                if not raw_data:
                    print("\nTimeout reading data")
                    break

                # Decode as ASCII
                try:
                    text_data = raw_data.decode("utf-8", errors="replace")
                except UnicodeDecodeError:
                    print("\nDecode error, skipping block")
                    continue

                # Parse samples
                gen1_samples, gen2_samples, partial_line, invalid = parse_ascii_data(
                    text_data, partial_line
                )
                total_invalid += invalid

                # Write samples to files
                for sample in gen1_samples:
                    f1.write(format_sample(sample, normalize) + "\n")
                for sample in gen2_samples:
                    f2.write(format_sample(sample, normalize) + "\n")

                gen1_count += len(gen1_samples)
                gen2_count += len(gen2_samples)

                # Periodic flush and progress
                if gen1_count % FLUSH_INTERVAL < len(gen1_samples):
                    f1.flush()
                    f2.flush()
                    elapsed = time.time() - start_time
                    rate = gen1_count / elapsed if elapsed > 0 else 0
                    progress = gen1_count * 100 / num_samples
                    sys.stdout.write(
                        f"\r{gen1_count:,} samples ({progress:.1f}%) "
                        f"at {rate:.0f} samples/s"
                    )
                    sys.stdout.flush()

        print()  # Newline after progress

    finally:
        ser.close()

    elapsed = time.time() - start_time
    print(f"Captured {gen1_count:,} Gen1 and {gen2_count:,} Gen2 samples in {elapsed:.1f}s")
    if total_invalid > 0:
        print(f"Skipped {total_invalid:,} invalid lines")

    return gen1_count, gen2_count


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Capture raw ADC samples from TrueRNG Pro (ASCII mode)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1_000_000,
        help="Number of samples to capture (default: 1000000)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Output normalized floats (0.0-1.0) instead of integers",
    )
    parser.add_argument(
        "--prefix",
        default="ascii_adc",
        help="Output file prefix (default: ascii_adc)",
    )
    parser.add_argument(
        "--port",
        help="Serial port (auto-detected if not specified)",
    )

    args = parser.parse_args()

    print("TrueRNG Pro ASCII ADC Capture")
    print("=" * 50)

    # Find device
    if args.port:
        port = args.port
        device_type = "TrueRNG (specified)"
    else:
        device = get_first_truerng()
        if device is None:
            print("No TrueRNG device found!")
            return 1
        port, device_type = device

    print(f"Using {device_type} on {port}")
    print("=" * 50)

    # Capture samples
    gen1, gen2 = capture_ascii_samples(
        port=port,
        num_samples=args.samples,
        output_prefix=args.prefix,
        normalize=args.normalize,
    )

    # Reset serial port
    reset_serial_port(port)

    if gen1 == 0:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
