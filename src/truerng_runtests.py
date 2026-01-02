#!/usr/bin/python3
"""Run statistical randomness tests on captured TrueRNG data files.

This script runs three statistical test suites on a data file:
- ent: Entropy analysis
- rngtest: FIPS 140-2 randomness tests
- dieharder: Comprehensive statistical test suite

Note: Dieharder needs 14GiB of data to not re-use (rewind) input data.
If you run this with less data, some dieharder results may be invalid.

Original author: Chris K Cockrum
Date: 6/14/2020
"""

import sys
from pathlib import Path

from truerng_utils import (
    check_test_binaries,
    run_dieharder,
    run_ent,
    run_rngtest,
)


def main() -> int:
    """Main entry point for running tests.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    # Platform check
    if sys.platform != "linux":
        print("Error: This script only runs on Linux.")
        return 1

    # Check for required binaries
    missing = check_test_binaries()
    if missing:
        print(f"Error: Missing required binaries: {', '.join(missing)}")
        return 1

    # Parse command line arguments
    if len(sys.argv) != 2:
        print("Usage: truerng_runtests.py FILENAME")
        return 1

    filename = Path(sys.argv[1])

    if not filename.is_file():
        print(f"{filename} not found")
        return 1

    # Print header
    print("=" * 50)
    print(f"TrueRNGpro Running Full Tests on {filename}")
    print("http://ubld.it")
    print("=" * 50)

    # Run all tests
    success = True
    success &= run_ent(filename)
    success &= run_rngtest(filename)
    success &= run_dieharder(filename)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
