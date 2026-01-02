#!/usr/bin/python3

# TrueRNG Finder
# Chris K Cockrum
# 6/8/2020
#
# Requires Python 3.8, pyserial
# On Linux - may need to be root or set /dev/tty port permissions to 666
#
# Python 3.8.xx is available here: https://www.python.org/downloads/
# Install Pyserial package with:   python -m pip install pyserial
# Install Pyusb package with:   python -m pip install pyusb

import sys
if sys.platform != "linux":
    sys.exit("Error: This script only runs on Linux.")

import serial
import usb
import subprocess
from serial.tools import list_ports


try:
    # Set com port to default None
    rng_com_port = None

    # Set mode to default None
    mode = None

    print("====================================================")
    print("= TrueRNG Finder                                   =")
    print("= for TrueRNG, TrueRNGV2, TrueRNGpro, TrueRNGproV2 =")
    print("= http://ubld.it                                   =")
    print("====================================================")

    ########################
    # Get list of TrueRNGs #
    ########################

    # Call list_ports to get com port info
    ports_available = list_ports.comports()

    # Loop on all available ports to find TrueRNG
    # Uses the first TrueRNG, TrueRNGpro, or TrueRNGproV2 found
    for temp in ports_available:
        #   print(temp[1] + ' : ' + temp[2])
        if "04D8:F5FE" in temp[2]:
            command = "lsusb -d 04d8:f5fe -v 2> /dev/null | grep bcdDevice"
            result = subprocess.check_output(command, shell=True)
            print(
                temp[0] + " : TrueRNG       : No SN      " + " : Rev " + str(result).split("  ")[-1].split("\\")[0]
            )
        if "16D0:0AA0" in temp[2]:
            command = "lsusb -d 16d0:0aa0 -v 2> /dev/null | grep bcdDevice"
            result = subprocess.check_output(command, shell=True)
            print(
                temp[0]
                + " : TrueRNGpro    : SN "
                + temp.serial_number
                + " : Rev "
                + str(result).split("  ")[-1].split("\\")[0]
            )
        if "04D8:EBB5" in temp[2]:
            command = "lsusb -d 04d8:ebb5 -v 2> /dev/null | grep bcdDevice"
            result = subprocess.check_output(command, shell=True)
            print(
                temp[0]
                + " : TrueRNGpro V2 : SN "
                + temp.serial_number
                + " : Rev "
                + str(result).split("  ")[-1].split("\\")[0]
            )

    print("====================================================")

except:
    print("Exiting now!")
