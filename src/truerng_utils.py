#!/usr/bin/python3
"""Shared utilities for TrueRNG device interaction.

This module provides common functionality for TrueRNG, TrueRNGpro, and TrueRNGproV2
hardware random number generators.
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path

import serial
from serial.tools import list_ports


# USB Vendor/Product IDs for device identification
TRUERNG_VID_PID = "04D8:F5FE"  # TrueRNG V1/V2/V3
TRUERNGPRO_VID_PID = "16D0:0AA0"  # TrueRNGpro V1
TRUERNGPROV2_VID_PID = "04D8:EBB5"  # TrueRNGpro V2

# Mode name to baudrate mapping
MODE_BAUDRATES: dict[str, int] = {
    "MODE_NORMAL": 300,  # Streams combined + Mersenne Twister
    "MODE_PSDEBUG": 1200,  # PS Voltage in mV in ASCII
    "MODE_RNGDEBUG": 2400,  # RNG Debug 0x0RRR 0x0RRR in ASCII
    "MODE_RNG1WHITE": 4800,  # RNG1 + Mersenne Twister
    "MODE_RNG2WHITE": 9600,  # RNG2 + Mersenne Twister
    "MODE_RAW_BIN": 19200,  # Raw ADC Samples in Binary Mode
    "MODE_RAW_ASC": 38400,  # Raw ADC Samples in Ascii Mode
    "MODE_UNWHITENED": 57600,  # Unwhitened RNG1-RNG2 (TrueRNGproV2 Only)
    "MODE_NORMAL_ASC": 115200,  # Normal in Ascii Mode (TrueRNGproV2 Only)
    "MODE_NORMAL_ASC_SLOW": 230400,  # Normal in Ascii Mode - Slow (TrueRNGproV2 Only)
}


def mode_change(mode: str, port: str, verbose: bool = False) -> bool:
    """Change the operating mode of a TrueRNGpro or TrueRNGproV2 device.

    Uses a "knock sequence" of baudrate changes to activate mode switching.
    Note: TrueRNG V1/V2/V3 devices do not support mode changes.

    Args:
        mode: The mode name (e.g., "MODE_NORMAL", "MODE_RAW_BIN").
        port: The serial port path (e.g., "/dev/ttyACM0").
        verbose: If True, print status messages.

    Returns:
        True if mode change was successful, False if mode was not recognized.

    Raises:
        serial.SerialException: If the serial port cannot be opened.
    """
    if mode not in MODE_BAUDRATES:
        if verbose:
            print(f"Mode not recognized: {mode}")
        return False

    target_baudrate = MODE_BAUDRATES[mode]

    # "Knock" sequence to activate mode change
    ser = serial.Serial(port=port, baudrate=110, timeout=1)
    time.sleep(0.5)
    ser.close()

    ser = serial.Serial(port=port, baudrate=300, timeout=1)
    ser.close()

    ser = serial.Serial(port=port, baudrate=110, timeout=1)
    ser.close()

    # Set target mode baudrate
    ser = serial.Serial(port=port, baudrate=target_baudrate, timeout=1)
    ser.close()

    if verbose:
        print(f"Switched to {mode}")

    return True


def find_truerng_devices() -> list[tuple[str, str]]:
    """Scan for connected TrueRNG devices.

    Returns:
        A list of tuples (port_path, device_type) for each detected device.
        Device types are: "TrueRNG", "TrueRNGpro", "TrueRNGproV2".
    """
    devices: list[tuple[str, str]] = []
    ports_available = list_ports.comports()

    for port_info in ports_available:
        hwid = port_info[2] if len(port_info) > 2 else ""

        if TRUERNG_VID_PID in hwid:
            devices.append((port_info[0], "TrueRNG"))
        elif TRUERNGPRO_VID_PID in hwid:
            devices.append((port_info[0], "TrueRNGpro"))
        elif TRUERNGPROV2_VID_PID in hwid:
            devices.append((port_info[0], "TrueRNGproV2"))

    return devices


def get_first_truerng() -> tuple[str, str] | None:
    """Get the first available TrueRNG device.

    Returns:
        A tuple (port_path, device_type) for the first detected device,
        or None if no devices are found.
    """
    devices = find_truerng_devices()
    return devices[0] if devices else None


def reset_serial_port(port: str) -> None:
    """Reset serial port settings on Linux.

    Pyserial may leave the port in a non-standard state; this resets it.
    Only has an effect on Linux systems.

    Args:
        port: The serial port path to reset.
    """
    if sys.platform == "linux":
        subprocess.run(["stty", "-F", port, "min", "1"], check=False)


def check_test_binaries() -> list[str]:
    """Check for required external test binaries.

    Returns:
        List of missing binary names.
    """
    required = ["ent", "dieharder", "rngtest"]
    return [binary for binary in required if shutil.which(binary) is None]


def run_ent(filename: Path) -> bool:
    """Run the ent entropy analysis tool.

    Args:
        filename: Path to the data file to analyze.

    Returns:
        True if successful, False otherwise.
    """
    output_file = filename.with_suffix(filename.suffix + ".ent.txt")
    print("\n *** Running ent *** \n")

    try:
        with open(output_file, "w") as outf:
            result = subprocess.run(
                ["ent", str(filename)],
                stdout=outf,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return result.returncode == 0
    except OSError as e:
        print(f"Failed to run ent: {e}")
        return False


def run_rngtest(filename: Path) -> bool:
    """Run the rngtest FIPS 140-2 tests.

    Args:
        filename: Path to the data file to analyze.

    Returns:
        True if successful, False otherwise.
    """
    output_file = filename.with_suffix(filename.suffix + ".rngtest.txt")
    print("\n *** Running rngtest *** \n")

    try:
        with open(filename, "rb") as inf, open(output_file, "w") as outf:
            result = subprocess.run(
                ["rngtest"],
                stdin=inf,
                stdout=outf,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return result.returncode == 0
    except OSError as e:
        print(f"Failed to run rngtest: {e}")
        return False


def run_dieharder(filename: Path) -> bool:
    """Run the dieharder statistical test suite.

    Args:
        filename: Path to the data file to analyze.

    Returns:
        True if successful, False otherwise.
    """
    output_file = filename.with_suffix(filename.suffix + ".dieharder.txt")
    print("\n *** Running dieharder *** \n")

    dieharder_args = [
        "dieharder",
        "-a",
        "-g",
        "201",
        "-s",
        "1",
        "-k",
        "2",
        "-Y",
        "1",
        "-f",
        str(filename),
    ]

    try:
        with open(output_file, "w") as outf:
            result = subprocess.run(
                dieharder_args,
                stdout=outf,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return result.returncode == 0
    except OSError as e:
        print(f"Failed to run dieharder: {e}")
        return False
