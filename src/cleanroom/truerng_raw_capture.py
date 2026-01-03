#!/usr/bin/python3
"""Capture raw ADC samples from TrueRNG Pro in RAW binary mode.

This script reads raw 10-bit ADC samples from the TrueRNG Pro device,
parsing the MODE_RAW_BIN format according to the specification.
Samples from each generator are saved to separate files.

RAW Binary Format (4 bytes per 2 samples):
  Byte 0: [SEQ=00][X:2][SAMPLE1_HI:4] - Gen1 high bits (9:6)
  Byte 1: [SEQ=01][SAMPLE1_LO:6]      - Gen1 low bits (5:0)
  Byte 2: [SEQ=10][X:2][SAMPLE2_HI:4] - Gen2 high bits (9:6)
  Byte 3: [SEQ=11][SAMPLE2_LO:6]      - Gen2 low bits (5:0)

Usage:
  python truerng_raw_capture.py [options]

Options:
  --samples N      Number of samples to capture (default: 1000000)
  --normalize      Output normalized floats (0.0-1.0) instead of integers
  --prefix PREFIX  Output file prefix (default: "raw_adc")
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

# RAW binary mode SEQ code masks
SEQ_MASK = 0xC0
SEQ_GEN1_HI = 0x00
SEQ_GEN1_LO = 0x40
SEQ_GEN2_HI = 0x80
SEQ_GEN2_LO = 0xC0

# Data masks for sample extraction
SAMPLE_HI_MASK = 0x0F  # Bits 3:0 contain sample bits 9:6
SAMPLE_LO_MASK = 0x3F  # Bits 5:0 contain sample bits 5:0

# ADC resolution
ADC_MAX_VALUE = 1023  # 10-bit ADC: 2^10 - 1

# Capture configuration
BLOCK_SIZE = 64 * 1024  # Read 64KB at a time
FLUSH_INTERVAL = 10000  # Flush output every N samples


def find_packet_boundary(data: bytes) -> int | None:
    """Find the start of a valid 4-byte packet in the data stream.

    Scans for a position where all 4 SEQ codes match the expected pattern.

    Args:
        data: Raw bytes from the device.

    Returns:
        Offset to the first valid packet, or None if not found.
    """
    for i in range(min(len(data) - 3, 256)):
        if (
            (data[i] & SEQ_MASK) == SEQ_GEN1_HI
            and (data[i + 1] & SEQ_MASK) == SEQ_GEN1_LO
            and (data[i + 2] & SEQ_MASK) == SEQ_GEN2_HI
            and (data[i + 3] & SEQ_MASK) == SEQ_GEN2_LO
        ):
            return i
    return None


def parse_raw_binary_samples(
    data: bytes,
    start_offset: int = 0,
) -> tuple[list[int], list[int], int]:
    """Parse raw binary data into Gen1 and Gen2 samples.

    Args:
        data: Raw bytes from MODE_RAW_BIN.
        start_offset: Byte offset to start parsing from.

    Returns:
        Tuple of (gen1_samples, gen2_samples, bytes_consumed).
    """
    gen1_samples: list[int] = []
    gen2_samples: list[int] = []
    offset = start_offset

    while offset + 4 <= len(data):
        g1_hi = data[offset]
        g1_lo = data[offset + 1]
        g2_hi = data[offset + 2]
        g2_lo = data[offset + 3]

        # Validate SEQ codes
        if (
            (g1_hi & SEQ_MASK) != SEQ_GEN1_HI
            or (g1_lo & SEQ_MASK) != SEQ_GEN1_LO
            or (g2_hi & SEQ_MASK) != SEQ_GEN2_HI
            or (g2_lo & SEQ_MASK) != SEQ_GEN2_LO
        ):
            # Sync lost - try to find next valid packet
            new_boundary = find_packet_boundary(data[offset:])
            if new_boundary is not None:
                offset += new_boundary
                continue
            break

        # Extract 10-bit samples: (high_bits << 6) | low_bits
        gen1 = ((g1_hi & SAMPLE_HI_MASK) << 6) | (g1_lo & SAMPLE_LO_MASK)
        gen2 = ((g2_hi & SAMPLE_HI_MASK) << 6) | (g2_lo & SAMPLE_LO_MASK)

        gen1_samples.append(gen1)
        gen2_samples.append(gen2)
        offset += 4

    return gen1_samples, gen2_samples, offset - start_offset


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


def capture_raw_samples(
    port: str,
    num_samples: int,
    output_prefix: str,
    normalize: bool,
) -> tuple[int, int]:
    """Capture raw ADC samples and save to files.

    Args:
        port: Serial port path.
        num_samples: Target number of samples to capture.
        output_prefix: Prefix for output filenames.
        normalize: If True, output normalized floats.

    Returns:
        Tuple of (gen1_count, gen2_count) samples captured.
    """
    # Switch to RAW binary mode
    if not mode_change("MODE_RAW_BIN", port, verbose=True):
        print("Failed to switch to MODE_RAW_BIN")
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
    sync_found = False
    leftover = b""

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

                # Prepend any leftover bytes from previous block
                data = leftover + raw_data

                # Find initial sync if needed
                if not sync_found:
                    boundary = find_packet_boundary(data)
                    if boundary is None:
                        print("Could not find packet boundary, retrying...")
                        leftover = data[-4:] if len(data) >= 4 else data
                        continue
                    data = data[boundary:]
                    sync_found = True
                    print(f"Synchronized at offset {boundary}")

                # Parse samples
                gen1_samples, gen2_samples, consumed = parse_raw_binary_samples(data)

                # Save leftover bytes for next iteration
                leftover = data[consumed:]

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

    return gen1_count, gen2_count


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Capture raw ADC samples from TrueRNG Pro"
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
        default="raw_adc",
        help="Output file prefix (default: raw_adc)",
    )
    parser.add_argument(
        "--port",
        help="Serial port (auto-detected if not specified)",
    )

    args = parser.parse_args()

    print("TrueRNG Pro Raw ADC Capture")
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
    gen1, gen2 = capture_raw_samples(
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
