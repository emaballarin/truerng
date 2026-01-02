#!/usr/bin/python3
"""Simple example for reading random data from a TrueRNG device.

This script demonstrates how to read random bytes from a TrueRNG device
and write them to a file. Configurable block size and loop count.

Original author: Chris K Cockrum
Date: 6/8/2020
"""

import sys
import time

import serial

from truerng_utils import (
    find_truerng_devices,
    mode_change,
    reset_serial_port,
)

# Configuration
BLOCK_SIZE = 102400  # Size of each read block in bytes
NUM_LOOPS = 10  # Number of blocks to read
OUTPUT_FILE = "random.bin"

# Set mode (only has effect on TrueRNGpro and TrueRNGproV2)
CAPTURE_MODE = "MODE_NORMAL"


def main() -> int:
    """Main entry point for data reading example.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    print("TrueRNGpro Data Read Example")
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

    if rng_com_port is None:
        print("No TrueRNG devices detected!")
        return 1

    print("=" * 50)
    print(f"Using com port:  {rng_com_port}")
    print(f"Block Size:      {BLOCK_SIZE / 1000:.2f} KB")
    print(f"Number of loops: {NUM_LOOPS}")
    print(f"Total size:      {BLOCK_SIZE * NUM_LOOPS / 1_000_000:.2f} MB")
    print(f"Writing to:      {OUTPUT_FILE}")
    print(f"Capture Mode:    {CAPTURE_MODE}")
    print("=" * 50)

    # Change to capture mode
    try:
        mode_change(CAPTURE_MODE, rng_com_port)
    except serial.SerialException as e:
        print(f"Failed to change mode: {e}")
        return 1

    # Open output file and read data
    try:
        with open(OUTPUT_FILE, "wb") as fp:
            try:
                ser = serial.Serial(port=rng_com_port, timeout=10)
            except serial.SerialException as e:
                print("Port Not Usable!")
                print(f"Do you have permissions to read {rng_com_port}?")
                print(f"Error: {e}")
                return 1

            try:
                if not ser.isOpen():
                    ser.open()

                ser.setDTR(True)
                ser.flushInput()

                total_bytes = 0

                for _ in range(NUM_LOOPS):
                    try:
                        before = time.time()
                        data = ser.read(BLOCK_SIZE)
                        after = time.time()
                    except serial.SerialException as e:
                        print(f"Read Failed: {e}")
                        break

                    total_bytes += len(data)
                    fp.write(data)

                    # Calculate transfer rate
                    elapsed = after - before
                    if elapsed > 0:
                        rate = float(BLOCK_SIZE) / (elapsed * 1_000_000.0) * 8
                    else:
                        rate = 0.0

                    print(f"{total_bytes} Bytes Read at {rate:2.3f} Mbits/s")

            finally:
                ser.close()

    except OSError as e:
        print(f"Error opening output file: {e}")
        return 1

    # Reset serial port settings
    reset_serial_port(rng_com_port)

    return 0


if __name__ == "__main__":
    sys.exit(main())
