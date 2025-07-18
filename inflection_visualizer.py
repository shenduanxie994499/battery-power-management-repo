import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "smoothed_normalized_data")
blacklist = ['30.0mA5msec-0.2mA5msec.csv','30.0mA4msec-0.2mA6msec.csv','30.0mA3msec-0.2mA7msec.csv']
filelist = [f for f in os.listdir(data_dir) if f.endswith('.csv') and f not in blacklist]

#filelist = ['20.0mA1msec-0.2mA4msec.csv']

# extracts parameters from filename
def extract_parameters(file_name):
    """
    Extract high_current, on_time, low_current, off_time, start_hour, end_hour from file name.
    Example: '30mA1msec-0.2mA24msec0-24hour.csv' → 30.0, 1, 0.2, 24, 0, 24
    """
    match = re.search(r'(\d+(?:\.\d+)?)mA(\d+)msec-(\d+(?:\.\d+)?)mA(\d+)msec(?:([0-9]+)-([0-9]+)hour)?', file_name)
    if match:
        on_current = float(match.group(1))
        on_time = int(match.group(2))
        off_current = float(match.group(3))
        off_time = int(match.group(4))
        start_hour = int(match.group(5)) if match.group(5) else 0
        end_hour = int(match.group(6)) if match.group(6) else 0
        return on_current, on_time, off_current, off_time, start_hour, end_hour
    else:
        raise ValueError(f"Filename '{file_name}' does not match expected pattern.")


def find_cutoff(capacity, voltage, scan_start_frac=0.2, negative_run=600):
    d1 = np.gradient(voltage, capacity)
    d2 = np.gradient(d1, capacity)
    start_idx = int(scan_start_frac * len(capacity))
    for i in range(start_idx, len(d2) - negative_run):
        if np.all(d2[i:i + negative_run] < -0.00005):
            return i
    return -1  # fallback

# function to calculate average discharge current based on discharge waveform
def average_current(on_current, on_time, off_current, off_time):
    return (on_current * on_time + off_current * off_time) / (on_time + off_time)


#fit a polynomial fit to the cropped voltage discharge curve and write it into a dictionary
def fit_model(capacity, voltage, filename, polydeg=7):
    I1, T1, I2, T2, _, _ = extract_parameters(filename)
    DC = T1 / (T1 + T2)
    T = T1 + T2

    capacity = np.array(capacity)
    voltage = np.array(voltage)

    # Smooth voltage before trimming
    # try:
    #     voltage = savgol_filter(voltage, window_length=1551, polyorder=3)
    # except Exception as e:
    #     print(f"{filename} failed")

    flatten_index = int(0.07 * len(capacity))
    capacity = capacity[flatten_index:]
    voltage = voltage[flatten_index:]

    # Find cutoff point
    i = find_cutoff(capacity, voltage)

    # Plot curve and cutoff
    plt.plot(capacity, voltage, label=filename)
    plt.scatter(capacity[i], voltage[i], color='red', s=20, zorder=5)


    # return {
    #     "DC" : DC,
    #     "I (mA)" : I1,
    #     "T (ms)" : T,
    #     **{f"coef_{i}" : c for i,c in enumerate(coeffs_trimmed[::-1])}
    # }



plt.figure()
for file in filelist:
    file_path = os.path.join(data_dir, file)

    capacity = []
    voltage = []

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if not row or len(row) < 2:
                continue
            try:
                c = float(row[0])
                v = float(row[1])
                capacity.append(c)
                voltage.append(v)
            except:
                continue

    # Downsample here (e.g., take every 10th point)
    downsample_rate = 10
    capacity = capacity[::downsample_rate]
    voltage = voltage[::downsample_rate]

    fit_model(capacity, voltage, file)

plt.xlabel("Capacity (mAh)")
plt.ylabel("Voltage (V)")
plt.title("Voltage vs Capacity (Predicted Curve)")
# plt.xlim(0,15)
# plt.ylim(0.9,1)
plt.grid(True)
#plt.legend()
plt.tight_layout()
plt.show()





# import csv
# import os
# import matplotlib.pyplot as plt
# import numpy as np
# import re
# from collections import *
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.cross_decomposition import PLSRegression
# from scipy.signal import savgol_filter
# from scipy.optimize import curve_fit

# base_dir = os.path.dirname(os.path.abspath(__file__))
# data_dir = os.path.join(base_dir, "smoothed_normalized_data")
# blacklist = ['30.0mA5msec-0.2mA5msec.csv','30.0mA4msec-0.2mA6msec.csv','30.0mA3msec-0.2mA7msec.csv']

# # blacklist = ['30.0mA5msec-0.2mA5msec.csv',
# #              '30.0mA4msec-0.2mA6msec.csv',
# #              '30.0mA3msec-0.2mA7msec.csv',
# #              '20.0mA2msec-0.2mA8msec.csv',
# #              '40.0mA8msec-0.2mA12msec.csv']
# filelist = [f for f in os.listdir(data_dir) if f.endswith('.csv') and f not in blacklist]

# #filelist = ['20.0mA1msec-0.2mA4msec.csv']

# # extracts parameters from filename
# def extract_parameters(file_name):
#     """
#     Extract high_current, on_time, low_current, off_time, start_hour, end_hour from file name.
#     Example: '30mA1msec-0.2mA24msec0-24hour.csv' → 30.0, 1, 0.2, 24, 0, 24
#     """
#     match = re.search(r'(\d+(?:\.\d+)?)mA(\d+)msec-(\d+(?:\.\d+)?)mA(\d+)msec(?:([0-9]+)-([0-9]+)hour)?', file_name)
#     if match:
#         on_current = float(match.group(1))
#         on_time = int(match.group(2))
#         off_current = float(match.group(3))
#         off_time = int(match.group(4))
#         start_hour = int(match.group(5)) if match.group(5) else 0
#         end_hour = int(match.group(6)) if match.group(6) else 0
#         return on_current, on_time, off_current, off_time, start_hour, end_hour
#     else:
#         raise ValueError(f"Filename '{file_name}' does not match expected pattern.")


# def find_cutoff(capacity, voltage, scan_start_frac=0.2, negative_run=600):
#     d1 = np.gradient(voltage, capacity)
#     d2 = np.gradient(d1, capacity)
#     start_idx = int(scan_start_frac * len(capacity))
#     for i in range(start_idx, len(d2) - negative_run):
#         if np.all(d2[i:i + negative_run] < -0.00005):
#             return i
#     return -1  # fallback

# # function to calculate average discharge current based on discharge waveform
# def average_current(on_current, on_time, off_current, off_time):
#     return (on_current * on_time + off_current * off_time) / (on_time + off_time)


# #fit a polynomial fit to the cropped voltage discharge curve and write it into a dictionary
# def fit_model(capacity, voltage, filename, polydeg=7):
#     I1, T1, I2, T2, _, _ = extract_parameters(filename)
#     DC = T1 / (T1 + T2)
#     T = T1 + T2

#     capacity = np.array(capacity)
#     voltage = np.array(voltage)

#     # Smooth voltage before trimming
#     try:
#         voltage_arr = savgol_filter(voltage, window_length=1551, polyorder=3)
#     except Exception as e:
#         print(f"[SKIPPED] {filename}: smoothing failed in stage 1 → {e}")

#     flatten_index = int(0.07 * len(capacity))
#     capacity = capacity[flatten_index:]
#     voltage = voltage[flatten_index:]

#     # Find cutoff point
#     i = find_cutoff(capacity, voltage)

#     # Plot curve and cutoff
#     plt.plot(capacity, voltage, label=filename)
#     plt.scatter(capacity[i], voltage[i], color='red', s=20, zorder=5)

# plt.figure()
# for file in filelist:
#     file_path = os.path.join(data_dir, file)

#     capacity = []
#     voltage = []

#     with open(file_path, 'r') as f:
#         reader = csv.reader(f)
#         next(reader)  # skip header
#         for row in reader:
#             if not row or len(row) < 2:
#                 continue
#             try:
#                 c = float(row[0])
#                 v = float(row[1])
#                 capacity.append(c)
#                 voltage.append(v)
#             except:
#                 continue

#     # Downsample here (e.g., take every 10th point)
#     downsample_rate = 10
#     capacity = capacity[::downsample_rate]
#     voltage = voltage[::downsample_rate]

#     fit_model(capacity, voltage, file)

# plt.xlabel("Capacity (mAh)")
# plt.ylabel("Voltage (V)")
# plt.title("Voltage vs Capacity (Predicted Curve)")
# # plt.xlim(0,15)
# # plt.ylim(0.9,1)
# plt.grid(True)
# #plt.legend()
# plt.tight_layout()
# plt.show()
