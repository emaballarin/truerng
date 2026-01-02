#!/usr/bin/python3
"""Capture large data blocks from TrueRNG and run statistical tests.

This script reads 14GB of data from a TrueRNG device (takes ~9 hours on
TrueRNGpro/TrueRNGproV2) and runs comprehensive statistical tests:
- ent: Entropy analysis
- rngtest: FIPS 140-2 randomness tests
- dieharder: Comprehensive statistical test suite

Note: Dieharder needs 14GiB of data to not re-use (rewind) input data.

Original author: Chris K Cockrum
Date: 6/9/2020
"""

import sys
import time
from pathlib import Path
from typing import BinaryIO

import serial

from truerng_utils import (
    find_truerng_devices,
    mode_change,
    reset_serial_port,
    run_dieharder,
    run_ent,
    run_rngtest,
)

# Data capture configuration
NUM_LOOPS = 14 * 1024  # 14 GiB for Dieharder to not repeat data
BLOCK_SIZE = 1024 * 1024  # 1 MiB per read

# Set mode (only has effect on TrueRNGpro and TrueRNGproV2)
CAPTURE_MODE = "MODE_NORMAL"


def generate_filename() -> Path:
    """Generate a timestamped output filename.

    Returns:
        Path object for the output file.
    """
    datetime_string = time.strftime("%Y%m%d.%H%M%S")
    return Path(f"TrueRNGpro_{datetime_string}.data")


def capture_data(
    port: str,
    output_file: BinaryIO,
    num_loops: int = NUM_LOOPS,
    block_size: int = BLOCK_SIZE,
) -> int:
    """Capture random data from a TrueRNG device.

    Args:
        port: Serial port path.
        output_file: Open file handle to write data to.
        num_loops: Number of blocks to read.
        block_size: Size of each block in bytes.

    Returns:
        Total bytes captured.

    Raises:
        serial.SerialException: If serial port operations fail.
    """
    ser = serial.Serial(port=port, timeout=10)

    try:
        if not ser.isOpen():
            ser.open()

        ser.setDTR(True)
        ser.flushInput()

        total_bytes = 0

        for i in range(num_loops):
            try:
                before = time.time()
                data = ser.read(block_size)
                after = time.time()
            except serial.SerialException as e:
                print(f"\nRead failed at block {i + 1}: {e}")
                break

            total_bytes += len(data)
            output_file.write(data)

            # Calculate transfer rate
            elapsed = after - before
            if elapsed > 0:
                rate = float(block_size) / (elapsed * 1_000_000.0) * 8
            else:
                rate = 0.0

            # Display progress
            progress = (i + 1) * 100 / num_loops
            sys.stdout.write(f"\r{i + 1} of {num_loops} MiB ({progress:2.1f}%) Read at {rate:2.3f} Mbits/s")
            sys.stdout.flush()

        print()  # Newline after progress
        return total_bytes

    finally:
        ser.close()


def main() -> int:
    """Main entry point for full testing.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    # Platform check
    if sys.platform != "linux":
        print("Error: This script only runs on Linux.")
        return 1

    print("TrueRNGpro Full Testing")
    print("http://ubld.it")
    print("=" * 50)

    # Find TrueRNG devices
    devices = find_truerng_devices()
    rng_com_port: str | None = None

    print("Detected devices:")
    for port, device_type in devices:
        print(f"  Found {device_type} on {port}")
        if rng_com_port is None:
            rng_com_port = port

    # Override with command line argument if provided
    if len(sys.argv) == 2:
        rng_com_port = sys.argv[1]
        print(f"\nUsing com port: {rng_com_port} (from command line)")
    elif rng_com_port:
        print(f"\nUsing com port: {rng_com_port} (first detected)")
    else:
        print("No TrueRNG devices detected!")
        return 1

    print("=" * 50)

    # Generate output filename
    output_filename = generate_filename()

    # Print configuration
    print(f"Block Size:      {BLOCK_SIZE / 1024 / 1024:.2f} MiB")
    print(f"Number of loops: {NUM_LOOPS}")
    print(f"Total size:      {NUM_LOOPS / 1024:.2f} GiB")
    print(f"Writing to:      {output_filename}")
    print(f"Capture Mode:    {CAPTURE_MODE}")
    print("=" * 50)

    # Change to capture mode
    try:
        mode_change(CAPTURE_MODE, rng_com_port)
    except serial.SerialException as e:
        print(f"Failed to change mode: {e}")
        return 1

    # Capture data
    try:
        with open(output_filename, "wb") as fp:
            try:
                total_bytes = capture_data(rng_com_port, fp)
                print(f"Captured {total_bytes / 1024 / 1024:.2f} MiB")
            except serial.SerialException as e:
                print(f"Serial port error: {e}")
                print(f"Do you have permissions to read {rng_com_port}?")
                return 1
    except OSError as e:
        print(f"Error opening output file: {e}")
        return 1

    # Run statistical tests
    run_ent(output_filename)
    run_rngtest(output_filename)
    run_dieharder(output_filename)

    # Reset serial port settings
    reset_serial_port(rng_com_port)

    return 0


if __name__ == "__main__":
    sys.exit(main())
