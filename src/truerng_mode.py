#!/usr/bin/python3
"""Change operating mode on TrueRNGpro and TrueRNGproV2 devices.

This script allows switching between different operating modes on
TrueRNGpro devices and displays sample output data.

Usage: truerng_mode.py [PORT] [MODE]
Example: python3 truerng_mode.py /dev/ttyACM0 MODE_NORMAL

Original author: Chris K Cockrum
Date: 6/5/2020
"""

import sys
import time

import serial

from truerng_utils import (
    MODE_BAUDRATES,
    find_truerng_devices,
    mode_change,
    reset_serial_port,
)

# Default operating mode
DEFAULT_MODE = "MODE_RNGDEBUG"

# Sample block size for testing output
SAMPLE_BLOCK_SIZE = 100 * 1024


def print_available_modes() -> None:
    """Print list of available modes."""
    print("\nAvailable modes:")
    for mode_name, baudrate in MODE_BAUDRATES.items():
        print(f"  {mode_name}: baudrate {baudrate}")


def main() -> int:
    """Main entry point for mode change utility.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    print("=" * 46)
    print("= TrueRNG Mode Change                        =")
    print("= for TrueRNGpro and TrueRNGproV2            =")
    print("= http://ubld.it                             =")
    print("=" * 46)

    # Find TrueRNG devices
    devices = find_truerng_devices()
    rng_com_port: str | None = None
    device_mode: str | None = None

    for port, device_type in devices:
        print(f"{port} : {device_type}")

        if device_type == "TrueRNG":
            print("  (Mode changes not supported on TrueRNG V1/V2/V3)")
            if rng_com_port is None:
                rng_com_port = port
                device_mode = device_type
        else:
            if rng_com_port is None:
                rng_com_port = port
                device_mode = device_type

        # Check if command-line port matches
        if len(sys.argv) >= 2 and port == sys.argv[1]:
            rng_com_port = port
            device_mode = device_type

    if rng_com_port is None:
        print("No TrueRNG devices detected!")
        return 1

    print("=" * 46)
    print(f"Using {device_mode} on {rng_com_port}")

    # Get mode from command line or use default
    operating_mode = DEFAULT_MODE
    if len(sys.argv) == 3:
        operating_mode = sys.argv[2]

    # Check if mode change is supported
    if device_mode == "TrueRNG":
        print("Mode changes not supported on TrueRNG V1/V2/V3")
        return 0

    # Validate mode
    if operating_mode not in MODE_BAUDRATES:
        print(f"Unknown mode: {operating_mode}")
        print_available_modes()
        return 1

    # Change mode
    try:
        if mode_change(operating_mode, rng_com_port, verbose=True):
            print(f"Mode changed to {operating_mode}")
        else:
            print(f"Failed to change mode to {operating_mode}")
            return 1
    except serial.SerialException as e:
        print(f"Serial error during mode change: {e}")
        return 1

    # Read sample data
    try:
        ser = serial.Serial(port=rng_com_port, timeout=10)
    except serial.SerialException as e:
        print(f"Port Not Usable: {e}")
        print(f"Do you have permissions to read {rng_com_port}?")
        return 1

    try:
        if not ser.isOpen():
            ser.open()

        ser.setDTR(True)
        ser.flushInput()

        before = time.time()
        data = ser.read(SAMPLE_BLOCK_SIZE)
        after = time.time()

    except serial.SerialException as e:
        print(f"Read Failed: {e}")
        return 1
    finally:
        ser.close()

    # Calculate transfer rate
    elapsed = after - before
    if elapsed > 0:
        rate = float(SAMPLE_BLOCK_SIZE) / (elapsed * 1_000_000.0) * 8
    else:
        rate = 0.0

    print(f"{SAMPLE_BLOCK_SIZE} Bytes Read at {rate:2.3f} Mbits/s")
    print("Output Sample:")
    print(data[0:80])

    # Reset serial port settings
    reset_serial_port(rng_com_port)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nExiting now!")
        sys.exit(0)
