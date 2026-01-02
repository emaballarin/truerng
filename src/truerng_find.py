#!/usr/bin/python3
"""Scan and identify all connected TrueRNG devices.

This script enumerates serial ports and identifies TrueRNG devices,
displaying their serial numbers and firmware revisions.

Original author: Chris K Cockrum
Date: 6/8/2020
"""

import subprocess
import sys

if sys.platform != "linux":
    sys.exit("Error: This script only runs on Linux.")

from serial.tools import list_ports

from truerng_utils import (
    TRUERNG_VID_PID,
    TRUERNGPRO_VID_PID,
    TRUERNGPROV2_VID_PID,
)


def get_firmware_revision(vid_pid: str) -> str:
    """Get firmware revision from lsusb for a given USB VID:PID.

    Args:
        vid_pid: USB Vendor:Product ID (e.g., "04d8:f5fe").

    Returns:
        Firmware revision string, or "Unknown" on error.
    """
    command = f"lsusb -d {vid_pid.lower()} -v 2> /dev/null | grep bcdDevice"
    try:
        result = subprocess.check_output(command, shell=True)
        return str(result).split("  ")[-1].split("\\")[0]
    except subprocess.CalledProcessError:
        return "Unknown"


def main() -> int:
    """Main entry point for device finder.

    Returns:
        Exit code (0 for success).
    """
    print("=" * 52)
    print("= TrueRNG Finder                                   =")
    print("= for TrueRNG, TrueRNGV2, TrueRNGpro, TrueRNGproV2 =")
    print("= http://ubld.it                                   =")
    print("=" * 52)

    # Get list of available COM ports
    ports_available = list_ports.comports()
    found_devices = 0

    for port_info in ports_available:
        hwid = port_info[2] if len(port_info) > 2 else ""
        port = port_info[0]
        serial_num = port_info.serial_number or "None"

        if TRUERNG_VID_PID in hwid:
            rev = get_firmware_revision("04d8:f5fe")
            print(f"{port} : TrueRNG       : No SN       : Rev {rev}")
            found_devices += 1

        elif TRUERNGPRO_VID_PID in hwid:
            rev = get_firmware_revision("16d0:0aa0")
            print(f"{port} : TrueRNGpro    : SN {serial_num} : Rev {rev}")
            found_devices += 1

        elif TRUERNGPROV2_VID_PID in hwid:
            rev = get_firmware_revision("04d8:ebb5")
            print(f"{port} : TrueRNGpro V2 : SN {serial_num} : Rev {rev}")
            found_devices += 1

    print("=" * 52)

    if found_devices == 0:
        print("No TrueRNG devices detected.")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nExiting now!")
        sys.exit(0)
