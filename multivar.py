import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import re
from collections import *

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

# function to calculate average discharge current based on discharge waveform
def average_current(on_current, on_time, off_current, off_time):
    return (on_current * on_time + off_current * off_time) / (on_time + off_time)

# Groups files that are continuations of one another (like 0-24 and 24-48 hour)
# Use a dictionary to do so, where the keys is the filename without the hours 
# and the value is a list of tuples in the form of (starthour, 'specific filename')
# The list of tuples is sorted in chronological order
def group_files(file_list):
    groups = defaultdict(list)

    for file in file_list:
        try:
            I1, T1, I2, T2, start, end = extract_parameters(file)
            key = f"{int(I1)}mA{T1}msec-{I2}mA{T2}msec"
            groups[key].append((start,file))
        except ValueError:
            print("Matching failed for: " + file)
        
    for key in groups.keys():
        groups[key].sort(key=lambda x: x[0])

    return groups

def linear_model(Q,a,b):
    return a * Q + b

def exponential_model(Q, A, k, c, offset):
    return A * np.exp(k * Q + c) + offset

def piecewise_model(capacity, voltage, polydeg = 5, slope_threshold = 0.1, dfrac= 0.1):
    mask = (voltage > 2.2) 
    capacity = capacity[mask]
    voltage = voltage[mask] 

    # Fit polynomial and compute derivative
    coeffs = np.polyfit(capacity, voltage, deg=5)
    poly = np.poly1d(coeffs)
    dpoly = poly.deriv()
    dV = dpoly(capacity)

    # Flattening detection
    epsilon = 0.005
    N = 10
    flatten_index = None
    for i in range(len(dV) - N):
        if np.all(np.abs(dV[i:i+N]) < epsilon):
            flatten_index = i
            break

    capacity = capacity[flatten_index:]
    voltage = voltage[flatten_index:]

    coeffs_trimmed = np.polyfit(capacity, voltage, deg=polydeg)
    poly_trimmed = np.poly1d(coeffs_trimmed)
    voltage_fit = poly_trimmed(capacity)

    return capacity, voltage, voltage_fit


def fit(path):
    file_list = [f for f in os.listdir(path) if f.endswith('.csv')]
    file_groups = group_files(file_list)

    for key, group in file_groups.items():
        I1, T1, I2, T2, _, _ = extract_parameters(group[0][1])
        DC = T1/(T1 + T2)
        I_avg = average_current(I1, T1, I2, T2)
        T = T1 + T2

        time_offset = 0
        time = []
        voltage = []
        for start, name in group:
            with open(name,'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if not any(row):
                        continue  # skip empty rows

                    # Extract sample interval
                    if not data_started and "Sample interval:" in row[0]:
                        try:
                            sample_interval = float(row[0].split(":")[1].strip())
                        except:
                            print("Sample interval not retrieved")
                            pass

                    # Detect start of data
                    if row[0].startswith("Enable:"):
                        data_started = True
                        continue 

                    # Parse actual data rows
                    if data_started:
                        try:
                            t = float(row[0]) * sample_interval + time_offset
                            v = float(row[1])
                            time.append(t / 3600)
                            voltage.append(v)
                        except:
                            continue

            _,_,_,_,start,end = extract_parameters(name)
            time_offset = (end - start) * 3600 #offset the timescale for the next sequence of data
            time_arr = np.array(time)
            voltage_arr = np.array(voltage)
            capacity_arr = I_avg * time_arr

            def model(Q,a,b,c,d,e):
                return a + b * Q + c * np.exp(d*x + e)

            











    
    
