#!/usr/bin/python3
"""Plot histograms of raw ADC samples from TrueRNG Pro.

This script reads raw 10-bit ADC samples from the TrueRNG Pro device
in MODE_RAW_BIN and plots the distribution histograms for each generator.

RAW Binary Format (4 bytes per 2 samples):
  Byte 0: [SEQ=00][X:2][SAMPLE1_HI:4] - Gen1 high bits (9:6)
  Byte 1: [SEQ=01][SAMPLE1_LO:6]      - Gen1 low bits (5:0)
  Byte 2: [SEQ=10][X:2][SAMPLE2_HI:4] - Gen2 high bits (9:6)
  Byte 3: [SEQ=11][SAMPLE2_LO:6]      - Gen2 low bits (5:0)

Usage:
  python truerng_raw_histogram.py [options]

Options:
  --samples N    Number of samples to collect (default: 100000)
  --bins N       Number of histogram bins (default: 256)
  --normalize    Show normalized values (0.0-1.0) on x-axis
  --port PORT    Serial port (auto-detected if not specified)
  --save FILE    Save plot to file instead of displaying
"""

import argparse
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
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


def collect_samples(port: str, num_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Collect raw ADC samples from the device.

    Args:
        port: Serial port path.
        num_samples: Target number of samples to collect.

    Returns:
        Tuple of (gen1_array, gen2_array) as numpy arrays.
    """
    # Switch to RAW binary mode
    if not mode_change("MODE_RAW_BIN", port, verbose=True):
        print("Failed to switch to MODE_RAW_BIN")
        return np.array([]), np.array([])

    # Open serial port
    try:
        ser = serial.Serial(port=port, timeout=10)
    except serial.SerialException as e:
        print(f"Failed to open port {port}: {e}")
        return np.array([]), np.array([])

    gen1_samples: list[int] = []
    gen2_samples: list[int] = []
    sync_found = False
    leftover = b""

    try:
        if not ser.isOpen():
            ser.open()

        ser.setDTR(True)
        ser.flushInput()

        print(f"Collecting {num_samples:,} samples...")
        start_time = time.time()

        while len(gen1_samples) < num_samples:
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
            g1, g2, consumed = parse_raw_binary_samples(data)
            gen1_samples.extend(g1)
            gen2_samples.extend(g2)

            # Save leftover bytes for next iteration
            leftover = data[consumed:]

            # Progress
            progress = len(gen1_samples) * 100 / num_samples
            sys.stdout.write(f"\r{len(gen1_samples):,} samples ({progress:.1f}%)")
            sys.stdout.flush()

        print()  # Newline after progress
        elapsed = time.time() - start_time
        print(f"Collected {len(gen1_samples):,} samples in {elapsed:.1f}s")

    finally:
        ser.close()

    # Trim to requested size
    gen1_samples = gen1_samples[:num_samples]
    gen2_samples = gen2_samples[:num_samples]

    return np.array(gen1_samples), np.array(gen2_samples)


def plot_histograms(
    gen1: np.ndarray,
    gen2: np.ndarray,
    num_bins: int,
    normalize: bool,
    save_path: str | None,
) -> None:
    """Plot side-by-side histograms for both generators.

    Args:
        gen1: Generator 1 samples.
        gen2: Generator 2 samples.
        num_bins: Number of histogram bins.
        normalize: If True, show normalized x-axis (0.0-1.0).
        save_path: If provided, save plot to this file.
    """
    if len(gen1) == 0 or len(gen2) == 0:
        print("No data to plot")
        return

    # Optionally normalize samples
    if normalize:
        gen1_plot = gen1 / ADC_MAX_VALUE
        gen2_plot = gen2 / ADC_MAX_VALUE
        x_label = "Normalized Value (0.0 - 1.0)"
        bin_range = (0.0, 1.0)
    else:
        gen1_plot = gen1
        gen2_plot = gen2
        x_label = "ADC Value (0 - 1023)"
        bin_range = (0, ADC_MAX_VALUE)

    # Calculate statistics
    gen1_mean = np.mean(gen1)
    gen1_std = np.std(gen1)
    gen2_mean = np.mean(gen2)
    gen2_std = np.std(gen2)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"TrueRNG Pro Raw ADC Distribution ({len(gen1):,} samples)",
        fontsize=14,
    )

    # Generator 1 histogram
    ax1.hist(gen1_plot, bins=num_bins, range=bin_range, color="steelblue", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax1.set_title("Generator 1")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Frequency")
    ax1.axvline(
        gen1_mean / ADC_MAX_VALUE if normalize else gen1_mean,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {gen1_mean:.1f}",
    )

    # Add statistics text box for Gen1
    stats_text1 = (
        f"Mean: {gen1_mean:.2f}\n"
        f"Std: {gen1_std:.2f}\n"
        f"Min: {np.min(gen1)}\n"
        f"Max: {np.max(gen1)}"
    )
    ax1.text(
        0.02,
        0.98,
        stats_text1,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Generator 2 histogram
    ax2.hist(gen2_plot, bins=num_bins, range=bin_range, color="coral", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax2.set_title("Generator 2")
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Frequency")
    ax2.axvline(
        gen2_mean / ADC_MAX_VALUE if normalize else gen2_mean,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {gen2_mean:.1f}",
    )

    # Add statistics text box for Gen2
    stats_text2 = (
        f"Mean: {gen2_mean:.2f}\n"
        f"Std: {gen2_std:.2f}\n"
        f"Min: {np.min(gen2)}\n"
        f"Max: {np.max(gen2)}"
    )
    ax2.text(
        0.02,
        0.98,
        stats_text2,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Plot histograms of raw ADC samples from TrueRNG Pro"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100_000,
        help="Number of samples to collect (default: 100000)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=256,
        help="Number of histogram bins (default: 256)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Show normalized values (0.0-1.0) on x-axis",
    )
    parser.add_argument(
        "--port",
        help="Serial port (auto-detected if not specified)",
    )
    parser.add_argument(
        "--save",
        metavar="FILE",
        help="Save plot to file instead of displaying",
    )

    args = parser.parse_args()

    print("TrueRNG Pro Raw ADC Histogram")
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

    # Collect samples
    gen1, gen2 = collect_samples(port, args.samples)

    # Reset serial port
    reset_serial_port(port)

    if len(gen1) == 0:
        return 1

    # Print summary statistics
    print("\nStatistics:")
    print(f"  Gen1: mean={np.mean(gen1):.2f}, std={np.std(gen1):.2f}, "
          f"min={np.min(gen1)}, max={np.max(gen1)}")
    print(f"  Gen2: mean={np.mean(gen2):.2f}, std={np.std(gen2):.2f}, "
          f"min={np.min(gen2)}, max={np.max(gen2)}")

    # Plot histograms
    plot_histograms(gen1, gen2, args.bins, args.normalize, args.save)

    return 0


if __name__ == "__main__":
    sys.exit(main())
