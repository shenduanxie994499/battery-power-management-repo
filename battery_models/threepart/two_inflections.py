import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from scipy.optimize import curve_fit

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = "./smoothed_normalized_data"

blacklist = ['30.0mA5msec-0.2mA5msec.csv',
             '30.0mA4msec-0.2mA6msec.csv',
             '30.0mA3msec-0.2mA7msec.csv',
             '20.0mA2msec-0.2mA8msec.csv',
             '40.0mA8msec-0.2mA12msec.csv']

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
    
def find_cutoff1(capacity, voltage, scan_start_frac=0.05, positive_run=9):
    d1 = np.gradient(voltage, capacity)
    d2 = np.gradient(d1, capacity)
    print(f"Max d2: {np.max(d2):.4e}, Min d2: {np.min(d2):.4e}")
    start_idx = int(scan_start_frac * len(capacity))
    for i in range(start_idx, len(d2) - positive_run):
        if np.all(d2[i:i + positive_run] < 0.000001):
            return i
    return -1  # fallback

def find_cutoff2(capacity, voltage, scan_start_frac=0.35, negative_run=10):
    d1 = np.gradient(voltage, capacity)
    d2 = np.gradient(d1, capacity)
    print(f"Max d2: {np.max(d2):.4e}, Min d2: {np.min(d2):.4e}")
    start_idx = int(scan_start_frac * len(capacity))
    for i in range(start_idx, len(d2) - negative_run):
        if np.all(d2[i:i + negative_run] < -0.000001):
            return i
    return -1  # fallback

# function to calculate average discharge current based on discharge waveform
def average_current(on_current, on_time, off_current, off_time):
    return (on_current * on_time + off_current * off_time) / (on_time + off_time)

#fit a polynomial fit to the cropped voltage discharge curve and write it into a dictionary
def fit_model(capacity, voltage, filename, downsample_rate=10):
    I1, T1, I2, T2, _, _ = extract_parameters(filename)
    DC = T1 / (T1 + T2)
    T = T1 + T2
    

    capacity = capacity[::downsample_rate]
    voltage = voltage[::downsample_rate]

    capacity = np.array(capacity)
    voltage = np.array(voltage)

    i = find_cutoff1(capacity,voltage)
    plt.scatter(capacity[i], voltage[i], color='blue', s=20, zorder=5)

    # Find first cutoff point
    j = find_cutoff2(capacity, voltage)
    plt.scatter(capacity[j], voltage[j], color='red', s=20, zorder=5)
    plt.plot(capacity, voltage, label=f"Data ({filename})")


plt.figure()
results = []
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

    result = fit_model(capacity, voltage, file)
    results.append(result)

plt.xlabel("Capacity (mAh)")
plt.ylabel("Voltage (V)")
plt.title("Voltage vs Capacity (Predicted Curve)")
plt.xlim(0,200)
plt.ylim(0.5,1)
plt.grid(True)
# plt.legend()
plt.tight_layout()
plt.show()
