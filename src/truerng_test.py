#!/usr/bin/python3
"""TrueRNG Series Testing utility with graphical output.

This script detects TrueRNG devices, runs comprehensive randomness tests,
and displays results with matplotlib graphs.

Supports:
- TrueRNG V1/V2/V3
- TrueRNGpro (V1)
- TrueRNGproV2

Original author: Chris K Cockrum
Date: 6/8/2020
"""

import math
import subprocess
import sys
import time

if sys.platform != "linux":
    sys.exit("Error: This script only runs on Linux.")

import matplotlib
import numpy as np
import serial
from matplotlib import pyplot
from serial.tools import list_ports

from truerng_utils import (
    TRUERNG_VID_PID,
    TRUERNGPRO_VID_PID,
    TRUERNGPROV2_VID_PID,
    mode_change,
    reset_serial_port,
)

print("=" * 52)
print("= TrueRNG Testing                                  =")
print("= for TrueRNG, TrueRNGV2, TrueRNGpro, TrueRNGproV2 =")
print("= http://ubld.it                                   =")
print("=" * 52)

# Test parameters for TrueRNG V1/V2/V3
TrueRNG_Min_Rate = 0.35  # Mbits/second
TrueRNG_Min_Entropy = 7.999  # bits/byte
TrueRNG_Max_Pi_Error = 1  # Maximum PI Error
TrueRNG_Max_Mean_Error = 0.5  # Maximum Mean Error
TrueRNG_Normal_Test_Size = 256 * 1024  # 0.25 Mbyte of data

# Test parameters for TrueRNGpro (V1)
TrueRNGpro_Min_Rate = 3.3  # Mbits/second
TrueRNGpro_Min_Entropy = 7.9995  # bits/byte
TrueRNGpro_Max_Pi_Error = 0.5  # Maximum PI Error
TrueRNGpro_Max_Mean_Error = 0.5  # Maximum Mean Error
TrueRNGpro_Normal_Test_Size = 1 * 1024 * 1024  # 1Mbyte of data
TrueRNGpro_Min_PS_Voltage = 7800  # Millivolts
TrueRNGpro_Max_PS_Voltage = 10300  # Millivolts
TrueRNGpro_Mean_Min = 450
TrueRNGpro_Mean_Max = 675
TrueRNGpro_Std_Min = 20
TrueRNGpro_Std_Max = 330

# Test parameters for TrueRNGproV2
TrueRNGproV2_Min_Rate = 3.3  # Mbits/second
TrueRNGproV2_Min_Entropy = 7.9995  # bits/byte
TrueRNGproV2_Max_Pi_Error = 0.5  # Maximum PI Error
TrueRNGproV2_Max_Mean_Error = 0.5  # Maximum Mean Error
TrueRNGproV2_Normal_Test_Size = 1 * 1024 * 1024  # 1Mbyte of data
TrueRNGproV2_Min_PS_Voltage = 15000  # Millivolts
TrueRNGproV2_Max_PS_Voltage = 17000  # Millivolts
TrueRNGproV2_Mean_Min = 512 - 128
TrueRNGproV2_Mean_Max = 512 + 128
TrueRNGproV2_Std_Min = 50
TrueRNGproV2_Std_Max = 150
TrueRNGproV2_W_Mean_Min = 256 - 32
TrueRNGproV2_W_Mean_Max = 256 + 32
TrueRNGproV2_W_Std_Min = 20
TrueRNGproV2_W_Std_Max = 90

# Create output file flag
output_file = False

# Define test failed flag
test_failed = False


def ps_voltage_test(comport: str) -> list[int]:
    """Test the power supply voltage on TrueRNGpro V1 and V2.

    Args:
        comport: Serial port path.

    Returns:
        List of voltage readings.
    """
    global test_failed
    mode_change("MODE_PSDEBUG", comport)

    try:
        ser = serial.Serial(port=comport, timeout=10)
    except serial.SerialException as e:
        print(f"*** Port Not Usable: {e}")
        print(f"*** Do you have permissions set to read {comport}?")
        return []

    try:
        if not ser.isOpen():
            ser.open()
        ser.setDTR(True)
        ser.flushInput()

        x = ser.read(6 * 256)
    except serial.SerialException as e:
        print(f"*** Read Failed: {e}")
        return []
    finally:
        ser.close()

    voltage_list: list[int] = []
    try:
        raw_list = x.decode("utf-8").split("\n")
        for item in raw_list:
            try:
                voltage = int(item)
                if 1000 <= voltage <= 30000:
                    voltage_list.append(voltage)
            except ValueError:
                continue
    except UnicodeDecodeError:
        pass

    if not voltage_list:
        print("*** FAILED *** Could not read voltage data")
        test_failed = True
        return []

    average_voltage = sum(voltage_list) / len(voltage_list)
    if Min_PS_Voltage < average_voltage < Max_PS_Voltage:
        print(f"*** PASSED *** Power Supply Voltage = {average_voltage / 1000:2.2f} Volts")
    else:
        print(f"*** FAILED *** Power Supply Voltage = {average_voltage / 1000:2.2f} Volts")
        test_failed = True

    return voltage_list


def normal_mode_test(comport: str) -> list[float]:
    """Run normal mode test on TrueRNG device.

    Args:
        comport: Serial port path.

    Returns:
        Frequency distribution list.
    """
    global test_failed
    if mode != "TrueRNG":
        mode_change("MODE_NORMAL", comport)

    fp = None
    if output_file == 1:
        try:
            fp = open("random.bin", "wb")
        except OSError as e:
            print(f"Error Opening File: {e}")

    try:
        ser = serial.Serial(port=comport, timeout=10)
    except serial.SerialException as e:
        print(f"*** Port Not Usable: {e}")
        print(f"*** Do you have permissions set to read {comport}?")
        return []

    try:
        if not ser.isOpen():
            ser.open()

        ser.setDTR(True)
        ser.flushInput()

        before = time.time()
        x = ser.read(Normal_Test_Size)
        after = time.time()
    except serial.SerialException as e:
        print(f"*** Read Failed: {e}")
        return []
    finally:
        ser.close()

    samples = x
    lengthRead = len(x)

    if output_file == 1 and fp is not None:
        fp.write(x)
        fp.close()

    # Calculate the rate
    rate = float(lengthRead) / ((after - before) * 1_000_000.0) * 8

    # Check transfer rate
    if rate >= 1.0:
        rate_str = f"{rate:2.3f} Mbits/s"
    else:
        rate_str = f"{rate * 1000:2.3f} Kbits/s"

    if rate > Min_Rate:
        print(f"*** PASSED *** NORMAL Mode {len(x)} Bytes Read at {rate_str}")
    else:
        print(f"*** FAILED *** NORMAL Mode {len(x)} Bytes Read at {rate_str}")
        test_failed = True

    # Count frequency of each byte value
    freqList = [0] * 256
    for byte in x:
        freqList[byte] += 1

    # Calculate Shannon entropy
    ent = 0.0
    for b in range(256):
        freqList[b] = freqList[b] / lengthRead
        if freqList[b] > 0:
            ent = ent + freqList[b] * math.log(freqList[b], 2)

    if (-ent) > 7.99:
        print(f"*** PASSED *** NORMAL Mode Entropy: {-ent:2.6f} bits/byte")
    else:
        print(f"*** FAILED *** NORMAL Mode Entropy: {-ent} bits/byte")
        test_failed = True

    # Monte Carlo estimate of pi
    circle_points = 0
    square_points = 0
    incirc = (2.0**48 - 1) ** 2
    sumx = 0
    j = 0.0
    k = 0.0

    for i in range(0, (len(x) - 24), 12):
        j = (
            (samples[i])
            + (samples[i + 1] << 8)
            + (samples[i + 2] << 16)
            + (samples[i + 3] << 24)
            + (samples[i + 4] << 32)
            + (samples[i + 5] << 40)
        )
        k = (
            (samples[i + 6])
            + (samples[i + 7] << 8)
            + (samples[i + 8] << 16)
            + (samples[i + 9] << 24)
            + (samples[i + 10] << 32)
            + (samples[i + 11] << 40)
        )
        square_points = square_points + 1
        sumx = (
            samples[i]
            + samples[i + 1]
            + samples[i + 2]
            + samples[i + 3]
            + samples[i + 4]
            + samples[i + 5]
            + samples[i + 6]
            + samples[i + 7]
            + samples[i + 8]
            + samples[i + 9]
            + samples[i + 10]
            + samples[i + 11]
            + sumx
        )
        if ((j * j) + (k * k)) < incirc:
            circle_points = circle_points + 1

    calcpi = 4.0 * float(circle_points) / float(square_points)
    pierror = 100.0 * math.fabs(calcpi - math.pi) / math.pi

    meanvalue = sumx / square_points / 12
    if math.fabs(meanvalue - 127.5) < Max_Mean_Error:
        print(f"*** PASSED *** Mean is {meanvalue:1.3f} (127.500 = random)")
    else:
        print(f"*** FAILED *** Mean is {meanvalue:1.3f} (127.500 = random)")
        test_failed = True

    if pierror < Max_Pi_Error:
        print(f"*** PASSED *** Monte Carlo estimate of pi is {calcpi:1.6f} ({pierror:1.6f}% Error)")
    else:
        print(f"*** FAILED *** Monte Carlo estimate of pi is {calcpi:1.6f} ({pierror:1.6f}% Error)")
        test_failed = True

    return freqList


def raw_asc_mode_test(comport: str) -> list[int]:
    """Run raw ASCII mode test on TrueRNGpro devices.

    Args:
        comport: Serial port path.

    Returns:
        Frequency distribution list for both generators.
    """
    global test_failed
    mode_change("MODE_RAW_ASC", comport)

    try:
        ser = serial.Serial(port=comport, timeout=10)
    except serial.SerialException as e:
        print(f"*** Port Not Usable: {e}")
        print(f"*** Do you have permissions set to read {comport}?")
        return []

    try:
        if not ser.isOpen():
            ser.open()

        ser.setDTR(True)
        ser.flushInput()

        before = time.time()
        x = ser.read(Normal_Test_Size)
        after = time.time()
    except serial.SerialException as e:
        print(f"*** Read Failed: {e}")
        return []
    finally:
        ser.close()

    rate = float(Normal_Test_Size) / ((after - before) * 1_000_000.0) * 8
    print(f"*** PASSED *** RAW ASCII Mode {len(x)} Bytes Read at {rate:2.3f} Mbits/s")

    try:
        raw_asc_list = x.decode("utf-8").split("\n")
    except UnicodeDecodeError:
        print("*** FAILED *** Could not decode raw ASCII data")
        test_failed = True
        return [0] * 2048

    freqList = [0] * 2048
    sum_gen1 = 0.0
    sum_gen2 = 0.0
    gen1samples = [0] * len(raw_asc_list)
    gen2samples = [0] * len(raw_asc_list)
    valid_samples = 0

    for i in range(len(raw_asc_list)):
        try:
            temp = raw_asc_list[i].split(",")
            gen1 = int(temp[0])
            gen2 = int(temp[1])
            gen1samples[i] = gen1
            gen2samples[i] = gen2
            sum_gen1 += gen1
            sum_gen2 += gen2
            if not (0 <= gen1 <= 1023) or not (0 <= gen2 <= 1023):
                break
            freqList[gen1] += 1
            freqList[gen2 + 1024] += 1
            valid_samples += 1
        except (ValueError, IndexError):
            break

    if valid_samples == 0:
        print("*** FAILED *** No valid samples in raw ASCII mode")
        test_failed = True
        return freqList

    gen1_mean = sum_gen1 / len(raw_asc_list)
    gen2_mean = sum_gen2 / len(raw_asc_list)

    if Min_Mean < gen1_mean < Max_Mean:
        print(f"*** PASSED *** Gen1 Mean = {gen1_mean:3.2f}")
    else:
        print(f"*** FAILED *** Gen1 Mean = {gen1_mean:3.2f}")
        test_failed = True

    if Min_Mean < gen2_mean < Max_Mean:
        print(f"*** PASSED *** Gen2 Mean = {gen2_mean:3.2f}")
    else:
        print(f"*** FAILED *** Gen2 Mean = {gen2_mean:3.2f}")
        test_failed = True

    gen1std = np.std(gen1samples)
    gen2std = np.std(gen2samples)

    if Min_Std < gen1std < Max_Std:
        print(f"*** PASSED *** Gen1 Standard Deviation = {gen1std:3.2f}")
    else:
        print(f"*** FAILED *** Gen1 Standard Deviation = {gen1std:3.2f}")
        test_failed = True

    if Min_Std < gen2std < Max_Std:
        print(f"*** PASSED *** Gen2 Standard Deviation = {gen2std:3.2f}")
    else:
        print(f"*** FAILED *** Gen2 Standard Deviation = {gen2std:3.2f}")
        test_failed = True

    return freqList


def unwhitened_mode_test(comport: str) -> list[int]:
    """Run unwhitened mode test on TrueRNGproV2.

    Args:
        comport: Serial port path.

    Returns:
        Frequency distribution list.
    """
    global test_failed
    mode_change("MODE_UNWHITENED", comport)

    try:
        ser = serial.Serial(port=comport, timeout=10)
    except serial.SerialException as e:
        print(f"*** Port Not Usable: {e}")
        print(f"*** Do you have permissions set to read {comport}?")
        return []

    try:
        if not ser.isOpen():
            ser.open()

        ser.setDTR(True)
        ser.flushInput()

        before = time.time()
        k = ser.read(Normal_Test_Size)
        after = time.time()
    except serial.SerialException as e:
        print(f"*** Read Failed: {e}")
        return []
    finally:
        ser.close()

    rate = float(Normal_Test_Size) / ((after - before) * 1_000_000.0) * 8
    print(f"*** PASSED *** UNWHITENED Mode {len(k)} Bytes Read at {rate:2.3f} Mbits/s")

    try:
        whitened_list = k.decode("utf-8").split(",")
    except UnicodeDecodeError:
        print("*** FAILED *** Could not decode unwhitened data")
        test_failed = True
        return [0] * 512

    freqList = [0] * 512
    whitened_samples = [0] * len(whitened_list)
    whitened_sum = 0

    for i in range(len(whitened_list)):
        try:
            temp = int(whitened_list[i])
            whitened_samples[i] = temp
            whitened_sum += temp
            if not (0 <= temp <= 511):
                break
            freqList[temp] += 1
        except ValueError:
            break

    if len(whitened_list) == 0:
        print("*** FAILED *** No unwhitened samples")
        test_failed = True
        return freqList

    whitened_mean = whitened_sum / len(whitened_list)

    if TrueRNGproV2_W_Mean_Min < whitened_mean < TrueRNGproV2_W_Mean_Max:
        print(f"*** PASSED *** Whitened Mean = {whitened_mean:3.2f}")
    else:
        print(f"*** FAILED *** Whitened Mean = {whitened_mean:3.2f}")
        test_failed = True

    whitenedstd = np.std(whitened_samples)
    if TrueRNGproV2_W_Std_Min < whitenedstd < TrueRNGproV2_W_Std_Max:
        print(f"*** PASSED *** Whitened Standard Deviation = {whitenedstd:3.2f}")
    else:
        print(f"*** FAILED *** Whitened Standard Deviation = {whitenedstd:3.2f}")
        test_failed = True

    return freqList


def move_figure(f: pyplot.Figure, x: int, y: int) -> None:
    """Move figure's upper left corner to pixel (x, y).

    Args:
        f: Matplotlib figure object.
        x: X coordinate in pixels.
        y: Y coordinate in pixels.
    """
    backend = matplotlib.get_backend()
    manager = f.canvas.manager
    if manager is None or not hasattr(manager, "window"):
        return  # Non-interactive or unsupported backend
    if backend == "TkAgg":
        manager.window.wm_geometry(f"+{x}+{y}")
    elif backend == "WXAgg":
        manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        manager.window.move(x, y)


def get_firmware_revision(device_mode: str) -> str:
    """Get firmware revision from lsusb.

    Args:
        device_mode: Device type ("TrueRNG", "TrueRNGpro", or "TrueRNGproV2").

    Returns:
        Firmware revision string, or "Unknown" on error.
    """
    commands = {
        "TrueRNG": "lsusb -d 04d8:f5fe -v 2> /dev/null | grep bcdDevice",
        "TrueRNGpro": "lsusb -d 16d0:0aa0 -v 2> /dev/null | grep bcdDevice",
        "TrueRNGproV2": "lsusb -d 04d8:ebb5 -v 2> /dev/null | grep bcdDevice",
    }

    command = commands.get(device_mode)
    if not command:
        return "Unknown"

    try:
        result = subprocess.check_output(command, shell=True)
        return str(result).split("  ")[-1].split("\\")[0]
    except subprocess.CalledProcessError:
        return "Unknown"


# Main test loop
try:
    while True:
        fig = None
        test_failed = False
        rng_com_port: str | None = None
        mode: str | None = None
        serial_number = "None"

        # Get list of COM ports
        ports_available = list_ports.comports()

        # Find TrueRNG devices
        for temp in ports_available:
            hwid = temp.hwid

            if TRUERNG_VID_PID in hwid:
                print(f"{temp[0]} : TrueRNG")
                if rng_com_port is None:
                    rng_com_port = temp[0]
                    serial_number = temp.serial_number
                    mode = "TrueRNG"
                elif len(sys.argv) == 2 and temp[0] == sys.argv[1]:
                    rng_com_port = temp[0]
                    serial_number = temp.serial_number
                    mode = "TrueRNG"

            if TRUERNGPRO_VID_PID in hwid:
                print(f"{temp[0]} : TrueRNGpro")
                if rng_com_port is None:
                    rng_com_port = temp[0]
                    serial_number = temp.serial_number
                    mode = "TrueRNGpro"
                elif len(sys.argv) == 2 and temp[0] == sys.argv[1]:
                    rng_com_port = temp[0]
                    serial_number = temp.serial_number
                    mode = "TrueRNGpro"

            if TRUERNGPROV2_VID_PID in hwid:
                print(f"{temp[0]} : TrueRNGpro V2")
                if rng_com_port is None:
                    rng_com_port = temp[0]
                    serial_number = temp.serial_number
                    mode = "TrueRNGproV2"
                elif len(sys.argv) == 2 and temp[0] == sys.argv[1]:
                    rng_com_port = temp[0]
                    serial_number = temp.serial_number
                    mode = "TrueRNGproV2"

        if rng_com_port is None:
            print("No TrueRNG devices detected!")

        print("=" * 52)

        # Print port info
        if len(sys.argv) == 2:
            rng_com_port = sys.argv[1]
            print(f"Using {mode} on {rng_com_port} (from command line)")
        elif rng_com_port:
            print(f"Using {mode} on {rng_com_port} (first detected)")

        print(f"Serial Number: {serial_number}")
        print(f"Firmware Rev : {get_firmware_revision(mode)}")

        # Set defaults for current mode/device
        if mode == "TrueRNG":
            Min_Rate = TrueRNG_Min_Rate
            Min_Entropy = TrueRNG_Min_Entropy
            Max_Pi_Error = TrueRNG_Max_Pi_Error
            Max_Mean_Error = TrueRNG_Max_Mean_Error
            Normal_Test_Size = TrueRNG_Normal_Test_Size
        elif mode == "TrueRNGpro":
            Min_Rate = TrueRNGpro_Min_Rate
            Min_Entropy = TrueRNGpro_Min_Entropy
            Max_Pi_Error = TrueRNGpro_Max_Pi_Error
            Max_Mean_Error = TrueRNGpro_Max_Mean_Error
            Normal_Test_Size = TrueRNGpro_Normal_Test_Size
            Min_PS_Voltage = TrueRNGpro_Min_PS_Voltage
            Max_PS_Voltage = TrueRNGpro_Max_PS_Voltage
            Min_Mean = TrueRNGpro_Mean_Min
            Max_Mean = TrueRNGpro_Mean_Max
            Min_Std = TrueRNGpro_Std_Min
            Max_Std = TrueRNGpro_Std_Max
        elif mode == "TrueRNGproV2":
            Min_Rate = TrueRNGproV2_Min_Rate
            Min_Entropy = TrueRNGproV2_Min_Entropy
            Max_Pi_Error = TrueRNGproV2_Max_Pi_Error
            Max_Mean_Error = TrueRNGproV2_Max_Mean_Error
            Normal_Test_Size = TrueRNGproV2_Normal_Test_Size
            Min_PS_Voltage = TrueRNGproV2_Min_PS_Voltage
            Max_PS_Voltage = TrueRNGproV2_Max_PS_Voltage
            Min_Mean = TrueRNGproV2_Mean_Min
            Max_Mean = TrueRNGproV2_Mean_Max
            Min_Std = TrueRNGproV2_Std_Min
            Max_Std = TrueRNGproV2_Std_Max

        if rng_com_port is None:
            print("No TrueRNG devices detected")
        else:
            print("=" * 52)

            # Tests for TrueRNG V1/V2/V3
            if mode == "TrueRNG":
                normal_freq_list = normal_mode_test(rng_com_port)

                fig = pyplot.figure(figsize=(8, 4), dpi=100)
                fig.suptitle(f"TrueRNG V1/V2/V3 Performance Plots ({rng_com_port})", fontsize=16)
                plt1 = fig.add_subplot(111)
                plt1.bar(np.arange(len(normal_freq_list)), normal_freq_list, 1)
                plt1.set_xlim([0, len(normal_freq_list)])
                plt1.set_title("Normal Mode Frequency Distribution")

                if test_failed:
                    plt1.set_facecolor((1.0, 0.0, 0.0))
                else:
                    plt1.set_facecolor((1.0, 1.0, 1.0))

                move_figure(fig, 0, 0)
                pyplot.draw()
                pyplot.pause(1)
                input("Press enter for another test or Ctrl-C to end.")
                pyplot.close(fig)

            # Tests for TrueRNGpro (V1)
            elif mode == "TrueRNGpro":
                ps_voltage_list = ps_voltage_test(rng_com_port)
                normal_freq_list = normal_mode_test(rng_com_port)
                raw_asc_freq_list = raw_asc_mode_test(rng_com_port)

                fig = pyplot.figure(figsize=(11, 7), dpi=100)
                fig.suptitle(f"TrueRNGpro (V1) Performance Plots ({rng_com_port})", fontsize=16)

                plt1 = fig.add_subplot(221)
                plt1.plot(np.arange(len(ps_voltage_list)), ps_voltage_list, "-")
                plt1.set_xlim(0, len(ps_voltage_list))
                plt1.set_ylim(Min_PS_Voltage, Max_PS_Voltage)
                plt1.set_title("Power Supply Voltage")

                plt2 = fig.add_subplot(222)
                plt2.bar(np.arange(len(normal_freq_list)), normal_freq_list, 1)
                plt2.set_xlim([0, len(normal_freq_list)])
                plt2.set_title("Normal Mode Frequency Distribution")

                plt3 = fig.add_subplot(223)
                plt3.bar(np.arange(1023), raw_asc_freq_list[0:1023:1], 1)
                plt3.set_xlim([0, len(raw_asc_freq_list) / 2])
                plt3.set_ylim(0, Normal_Test_Size / 768)
                plt3.set_title("Generator 1 Raw ASCII Mode Frequency Distribution")

                plt4 = fig.add_subplot(224)
                plt4.bar(np.arange(1023), raw_asc_freq_list[1024:2047:1], 1)
                plt4.set_xlim([0, len(raw_asc_freq_list) / 2])
                plt4.set_ylim(0, Normal_Test_Size / 768)
                plt4.set_title("Generator 2 Raw ASCII Mode Frequency Distribution")

                if test_failed:
                    for plt in [plt1, plt2, plt3, plt4]:
                        plt.set_facecolor((1.0, 0.0, 0.0))
                else:
                    for plt in [plt1, plt2, plt3, plt4]:
                        plt.set_facecolor((1.0, 1.0, 1.0))

                move_figure(fig, 0, 0)
                pyplot.draw()
                pyplot.pause(1)
                input("Press enter for another test or Ctrl-C to end.")
                pyplot.close(fig)

            # Tests for TrueRNGproV2
            elif mode == "TrueRNGproV2":
                ps_voltage_list = ps_voltage_test(rng_com_port)
                normal_freq_list = normal_mode_test(rng_com_port)
                raw_asc_freq_list = raw_asc_mode_test(rng_com_port)
                unwhitened_freq_list = unwhitened_mode_test(rng_com_port)

                fig = pyplot.figure(figsize=(11, 9), dpi=100)
                fig.suptitle(f"TrueRNGproV2 Performance Plots ({rng_com_port})", fontsize=16)

                plt1 = fig.add_subplot(321)
                plt1.plot(np.arange(len(ps_voltage_list)), ps_voltage_list, "-")
                plt1.set_xlim(0, len(ps_voltage_list))
                plt1.set_ylim(Min_PS_Voltage, Max_PS_Voltage)
                plt1.set_title("Power Supply Voltage")

                plt2 = fig.add_subplot(322)
                plt2.bar(np.arange(len(normal_freq_list)), normal_freq_list, 1)
                plt2.set_xlim([0, len(normal_freq_list)])
                plt2.set_title("Normal Mode Frequency Distribution")

                plt3 = fig.add_subplot(323)
                plt3.bar(np.arange(1023), raw_asc_freq_list[0:1023:1], 1)
                plt3.set_xlim([0, len(raw_asc_freq_list) / 2])
                plt3.set_ylim(0, Normal_Test_Size / 1024)
                plt3.set_title("Generator 1 Raw ASCII Mode Frequency Distribution")

                plt4 = fig.add_subplot(324)
                plt4.bar(np.arange(1023), raw_asc_freq_list[1024:2047:1], 1)
                plt4.set_xlim([0, len(raw_asc_freq_list) / 2])
                plt4.set_ylim(0, Normal_Test_Size / 1024)
                plt4.set_title("Generator 2 Raw ASCII Mode Frequency Distribution")

                plt5 = fig.add_subplot(3, 2, (5, 6))
                plt5.bar(np.arange(len(unwhitened_freq_list)), unwhitened_freq_list, 1)
                plt5.set_xlim([0, len(unwhitened_freq_list)])
                plt5.set_ylim(0, Normal_Test_Size / 256)
                plt5.set_title("Unwhitened Mode Frequency Distribution")

                if test_failed:
                    for plt in [plt1, plt2, plt3, plt4, plt5]:
                        plt.set_facecolor((1.0, 0.0, 0.0))
                else:
                    for plt in [plt1, plt2, plt3, plt4, plt5]:
                        plt.set_facecolor((1.0, 1.0, 1.0))

                move_figure(fig, 0, 0)
                pyplot.draw()
                pyplot.pause(1)
                input("Press enter for another test or Ctrl-C to end.")
                pyplot.close(fig)

            print("=" * 52)
            print("=" * 17 + " NEW TEST " + "=" * 25)
            print("=" * 52)

except KeyboardInterrupt:
    if fig:
        pyplot.close(fig)
    print("\nExiting now!")

# Reset serial port settings
if rng_com_port:
    reset_serial_port(rng_com_port)
