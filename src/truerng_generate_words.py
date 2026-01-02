#!/usr/bin/python3
"""Generate random word lists using TrueRNG hardware random number generator.

This script reads random bytes from a TrueRNG device and generates a list of
random words from the NLTK English word corpus. These can be used as passphrases.

Original author: Chris K Cockrum
Date: 7/23/20
"""

import math
import sys

import nltk
import serial

from truerng_utils import (
    find_truerng_devices,
    mode_change,
    reset_serial_port,
)

# Number of random words to generate
NUMBER_OF_WORDS = 20

# Minimum and maximum word length for filtering
MIN_WORD_LENGTH = 3
MAX_WORD_LENGTH = 9

# Set mode (only has effect on TrueRNGpro and TrueRNGproV2)
CAPTURE_MODE = "MODE_NORMAL"


def load_wordlist() -> list[str]:
    """Load the English word list from NLTK corpus.

    Returns:
        List of English words.

    Raises:
        LookupError: If the NLTK words corpus is not installed.
    """
    try:
        return nltk.corpus.words.words()
    except LookupError:
        print("NLTK words corpus not found. Downloading...")
        nltk.download("words")
        return nltk.corpus.words.words()


def bytes_to_int(data: bytes, offset: int) -> int:
    """Convert 3 consecutive bytes to a 24-bit integer.

    Args:
        data: Byte array to read from.
        offset: Starting position in the array.

    Returns:
        24-bit integer value.
    """
    return data[offset] + (data[offset + 1] * 256) + (data[offset + 2] * 65536)


def generate_random_words(
    random_bytes: bytes,
    wordlist: list[str],
    count: int,
    min_len: int = MIN_WORD_LENGTH,
    max_len: int = MAX_WORD_LENGTH,
) -> list[str]:
    """Generate random words from a wordlist using random bytes.

    Args:
        random_bytes: Raw random bytes from the RNG device.
        wordlist: List of words to choose from.
        count: Number of words to generate.
        min_len: Minimum word length to accept.
        max_len: Maximum word length to accept.

    Returns:
        List of randomly selected words.

    Raises:
        ValueError: If not enough random bytes to generate requested words.
    """
    words: list[str] = []
    index = 0
    step = 3  # Use 3 bytes per word selection
    wordlist_length = len(wordlist)

    while len(words) < count:
        if index + 2 >= len(random_bytes):
            raise ValueError(f"Insufficient random bytes to generate word list. Got {len(words)} of {count} words.")

        # Convert 3 bytes to an integer
        random_value = bytes_to_int(random_bytes, index)
        index += step

        # Map to wordlist range
        word_index = random_value % wordlist_length
        word = wordlist[word_index]

        # Filter by word length
        if min_len < len(word) < max_len:
            words.append(word)

    return words


def main() -> None:
    """Main entry point for word list generation."""
    print("TrueRNGpro Random Word Generator")
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

    # Load wordlist
    try:
        wordlist = load_wordlist()
    except Exception as e:
        print(f"Failed to load word list: {e}")
        sys.exit(1)

    wordlist_length = len(wordlist)
    entropy_per_3_words = math.log(wordlist_length, 2) * 3

    print(f"Choosing 3 of these words would give a password with about {entropy_per_3_words:.1f} bits of entropy")
    print("=" * 50)

    # Generate and display random words
    try:
        words = generate_random_words(random_data, wordlist, NUMBER_OF_WORDS)
        for word in words:
            print(word)
    except ValueError as e:
        print(f"Word generation failed: {e}")
        sys.exit(1)

    # Reset serial port settings on Linux
    reset_serial_port(rng_com_port)


if __name__ == "__main__":
    main()
