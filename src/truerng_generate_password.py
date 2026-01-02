#!/usr/bin/python3
"""Generate cryptographic passwords using TrueRNG hardware random number generator.

This script reads random bytes from a TrueRNG device and generates two types
of random passwords: one using all printable ASCII characters, and one using
only alphanumeric characters.

Original author: Chris K Cockrum
Date: 7/23/20
"""

import sys
from collections.abc import Callable

import serial

from truerng_utils import (
    find_truerng_devices,
    mode_change,
    reset_serial_port,
)

# Number of random characters to generate
PASSWORD_LENGTH = 20

# Set mode (only has effect on TrueRNGpro and TrueRNGproV2)
CAPTURE_MODE = "MODE_NORMAL"


def is_printable_ascii(byte_value: int) -> bool:
    """Check if byte value represents a printable ASCII character (33-126)."""
    return 33 <= byte_value <= 126


def is_alphanumeric(byte_value: int) -> bool:
    """Check if byte value represents an alphanumeric ASCII character."""
    # 0-9: 48-57, A-Z: 65-90, a-z: 97-122
    return (48 <= byte_value <= 57) or (65 <= byte_value <= 90) or (97 <= byte_value <= 122)


def generate_password(random_bytes: bytes, length: int, filter_func: Callable[[int], bool]) -> str:
    """Generate a password from random bytes using a character filter.

    Args:
        random_bytes: Raw random bytes from the RNG device.
        length: Desired password length.
        filter_func: Function to filter valid characters (takes int, returns bool).

    Returns:
        Generated password string of the requested length.

    Raises:
        ValueError: If not enough valid characters in random_bytes.
    """
    password_chars: list[str] = []
    index = 0
    step = 3  # Skip bytes to reduce bias

    while len(password_chars) < length:
        if index >= len(random_bytes):
            raise ValueError(
                f"Insufficient random bytes to generate password. Got {len(password_chars)} of {length} characters."
            )

        byte_value = random_bytes[index]
        index += step

        if filter_func(byte_value):
            password_chars.append(chr(byte_value))

    return "".join(password_chars)


def main() -> None:
    """Main entry point for password generation."""
    print("TrueRNGpro Password Generator")
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
        sys.exit(1)

    print("=" * 50)
    print(f"Using com port:  {rng_com_port}")
    print(f"Capture Mode:    {CAPTURE_MODE}")
    print("=" * 50)

    # Change to specified mode (only has effect on TrueRNGpro and TrueRNGproV2)
    try:
        mode_change(CAPTURE_MODE, rng_com_port)
    except serial.SerialException as e:
        print(f"Failed to change mode: {e}")
        sys.exit(1)

    # Read random data from device
    try:
        ser = serial.Serial(port=rng_com_port, timeout=10)
    except serial.SerialException as e:
        print("Port Not Usable!")
        print(f"Do you have permissions set to read {rng_com_port}?")
        print(f"Error: {e}")
        sys.exit(1)

    try:
        if not ser.isOpen():
            ser.open()

        ser.setDTR(True)
        ser.flushInput()

        random_data = ser.read(2048)

    except serial.SerialException as e:
        print(f"Read Failed: {e}")
        sys.exit(1)
    finally:
        ser.close()

    # Generate and display passwords
    print("Entropy")
    print("Printable Characters are   : 6.58 bits/char")
    print("Letter / Number Characters : 5.95 bits/char")
    print("=" * 50)

    try:
        printable_password = generate_password(random_data, PASSWORD_LENGTH, is_printable_ascii)
        print(f"  Printable Characters Password: {printable_password}")

        alphanum_password = generate_password(random_data, PASSWORD_LENGTH, is_alphanumeric)
        print(f"Letters / Numbers Only Password: {alphanum_password}")

    except ValueError as e:
        print(f"Password generation failed: {e}")
        sys.exit(1)

    print("=" * 50)

    # Reset serial port settings on Linux
    reset_serial_port(rng_com_port)


if __name__ == "__main__":
    main()
