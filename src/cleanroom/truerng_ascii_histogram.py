#!/usr/bin/python3
"""Plot histograms of raw ADC samples from TrueRNG Pro (ASCII mode).

This script reads raw 10-bit ADC samples from the TrueRNG Pro device
in MODE_RAW_ASC and plots the distribution histograms for each generator.

RAW ASCII Format:
  Each line contains: gen1_value,gen2_value
  Values are 10-bit ADC readings (0-1023)

Usage:
  python truerng_ascii_histogram.py [options]

Options:
  --samples N    Number of samples to collect (default: 100000)
  --bins N       Number of histogram bins (default: 256)
  --normalize    Show normalized values (0.0-1.0) on x-axis
  --port PORT    Serial port (auto-detected if not specified)
  --save FILE    Save plot to file instead of displaying
  --legacy       Also compute stats using the buggy algorithm from truerng_test.py
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

# ADC resolution
ADC_MAX_VALUE = 1023  # 10-bit ADC: 2^10 - 1

# Capture configuration
BLOCK_SIZE = 64 * 1024  # Read 64KB at a time


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


def parse_ascii_data_legacy(data: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Parse ASCII data using the BUGGY algorithm from truerng_test.py.

    This reproduces the bugs in the original code for comparison:
    1. Pre-allocates arrays with zeros
    2. Breaks on any parse error (instead of skipping)
    3. Breaks on out-of-range values
    4. Returns arrays that may contain trailing zeros

    Args:
        data: String data from MODE_RAW_ASC.

    Returns:
        Tuple of (gen1_array, gen2_array, total_lines).
        Arrays are zero-padded to total_lines length.
    """
    lines = data.split("\n")
    total_lines = len(lines)

    # Bug #1: Pre-allocate arrays with zeros
    gen1samples = [0] * total_lines
    gen2samples = [0] * total_lines

    sum_gen1 = 0.0
    sum_gen2 = 0.0

    for i in range(total_lines):
        try:
            temp = lines[i].split(",")
            gen1 = int(temp[0])
            gen2 = int(temp[1])
            gen1samples[i] = gen1
            gen2samples[i] = gen2
            sum_gen1 += gen1
            sum_gen2 += gen2
            # Bug #3: Break on out-of-range values
            if not (0 <= gen1 <= 1023) or not (0 <= gen2 <= 1023):
                break
        except (ValueError, IndexError):
            # Bug #2: Break on any parse error
            break

    return np.array(gen1samples), np.array(gen2samples), total_lines


def compute_legacy_stats(
    gen1: np.ndarray,
    gen2: np.ndarray,
    total_lines: int,
) -> dict[str, float]:
    """Compute statistics using the BUGGY algorithm from truerng_test.py.

    Bugs reproduced:
    - Mean uses total_lines as denominator (not valid sample count)
    - Std is calculated on zero-padded arrays

    Args:
        gen1: Generator 1 samples (zero-padded).
        gen2: Generator 2 samples (zero-padded).
        total_lines: Total number of lines (used as denominator).

    Returns:
        Dictionary with legacy mean and std values.
    """
    # Bug #4: Divide by total lines, not valid samples
    gen1_mean = float(np.sum(gen1)) / total_lines
    gen2_mean = float(np.sum(gen2)) / total_lines

    # Bug #5: Std on zero-padded array
    gen1_std = float(np.std(gen1))
    gen2_std = float(np.std(gen2))

    return {
        "gen1_mean": gen1_mean,
        "gen2_mean": gen2_mean,
        "gen1_std": gen1_std,
        "gen2_std": gen2_std,
    }


def collect_samples(
    port: str,
    num_samples: int,
    keep_raw: bool = False,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Collect ASCII mode ADC samples from the device.

    Args:
        port: Serial port path.
        num_samples: Target number of samples to collect.
        keep_raw: If True, also return raw text data for legacy processing.

    Returns:
        Tuple of (gen1_array, gen2_array, raw_text).
        raw_text is empty string if keep_raw is False.
    """
    # Switch to RAW ASCII mode
    if not mode_change("MODE_RAW_ASC", port, verbose=True):
        print("Failed to switch to MODE_RAW_ASC")
        return np.array([]), np.array([]), ""

    # Open serial port
    try:
        ser = serial.Serial(port=port, timeout=10)
    except serial.SerialException as e:
        print(f"Failed to open port {port}: {e}")
        return np.array([]), np.array([]), ""

    gen1_samples: list[int] = []
    gen2_samples: list[int] = []
    total_invalid = 0
    partial_line = ""
    raw_text_chunks: list[str] = [] if keep_raw else []

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

            # Decode as ASCII
            try:
                text_data = raw_data.decode("utf-8", errors="replace")
            except UnicodeDecodeError:
                print("\nDecode error, skipping block")
                continue

            # Keep raw text for legacy processing if requested
            if keep_raw:
                raw_text_chunks.append(text_data)

            # Parse samples
            g1, g2, partial_line, invalid = parse_ascii_data(text_data, partial_line)
            gen1_samples.extend(g1)
            gen2_samples.extend(g2)
            total_invalid += invalid

            # Progress
            progress = len(gen1_samples) * 100 / num_samples
            sys.stdout.write(f"\r{len(gen1_samples):,} samples ({progress:.1f}%)")
            sys.stdout.flush()

        print()  # Newline after progress
        elapsed = time.time() - start_time
        print(f"Collected {len(gen1_samples):,} samples in {elapsed:.1f}s")
        if total_invalid > 0:
            print(f"Skipped {total_invalid:,} invalid lines")

    finally:
        ser.close()

    # Trim to requested size
    gen1_samples = gen1_samples[:num_samples]
    gen2_samples = gen2_samples[:num_samples]

    # Combine raw text chunks
    raw_text = "".join(raw_text_chunks) if keep_raw else ""

    return np.array(gen1_samples), np.array(gen2_samples), raw_text


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
        f"TrueRNG Pro ASCII ADC Distribution ({len(gen1):,} samples)",
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
        description="Plot histograms of raw ADC samples from TrueRNG Pro (ASCII mode)"
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
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Also compute stats using the buggy algorithm from truerng_test.py",
    )

    args = parser.parse_args()

    print("TrueRNG Pro ASCII ADC Histogram")
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
    gen1, gen2, raw_text = collect_samples(port, args.samples, keep_raw=args.legacy)

    # Reset serial port
    reset_serial_port(port)

    if len(gen1) == 0:
        return 1

    # Print summary statistics (correct algorithm)
    print("\nStatistics (correct algorithm):")
    print(f"  Gen1: mean={np.mean(gen1):.2f}, std={np.std(gen1):.2f}, "
          f"min={np.min(gen1)}, max={np.max(gen1)}")
    print(f"  Gen2: mean={np.mean(gen2):.2f}, std={np.std(gen2):.2f}, "
          f"min={np.min(gen2)}, max={np.max(gen2)}")

    # Print legacy statistics if requested
    if args.legacy and raw_text:
        print("\nStatistics (LEGACY buggy algorithm from truerng_test.py):")
        legacy_gen1, legacy_gen2, total_lines = parse_ascii_data_legacy(raw_text)
        legacy_stats = compute_legacy_stats(legacy_gen1, legacy_gen2, total_lines)
        print(f"  Gen1: mean={legacy_stats['gen1_mean']:.2f}, std={legacy_stats['gen1_std']:.2f}")
        print(f"  Gen2: mean={legacy_stats['gen2_mean']:.2f}, std={legacy_stats['gen2_std']:.2f}")
        print(f"  (Total lines: {total_lines}, zeros in arrays bias these values)")

        # Show expected ranges for comparison
        print("\nOriginal spec ranges (for reference):")
        print("  Mean: 450-550")
        print("  Std:  20-180")

    # Plot histograms
    plot_histograms(gen1, gen2, args.bins, args.normalize, args.save)

    return 0


if __name__ == "__main__":
    sys.exit(main())
