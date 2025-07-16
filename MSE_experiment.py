import os
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import *
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error 
import csv

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "oddballs/norm_smooth")
filelist = ['30.0mA5msec-0.2mA5msec.csv', '30.0mA4msec-0.2mA6msec.csv','30.0mA3msec-0.2mA7msec.csv','20.0mA2msec-0.2mA8msec.csv']


# data_dir = os.path.join(base_dir, "smoothed_normalized_data")
# filelist = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

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

def linear(x,a,b):
    return a * x + b

def exponential_shifted(x, c, d,x0):
    return c * (np.exp(d * x) - np.exp(d * x0))

#fit a polynomial fit to the cropped voltage discharge curve and write it into a dictionary
def eval_model(capacity, voltage, filename, cutoff, downsample_rate=10):
    I1, T1, I2, T2, _, _ = extract_parameters(filename)
    DC = T1 / (T1 + T2)
    T = T1 + T2

    # Find cutoff point
    i = cutoff

    x_lin = capacity[:i]
    y_lin = voltage[:i]
    x_exp = capacity[i:]
    y_exp = voltage[i:]

    params_lin, _ = curve_fit(linear,x_lin,y_lin)
    a,b = params_lin

    y_exp_residual = y_exp - linear(x_exp, a, b)
    x0 = x_exp[0]

    def exponential(x,c,d):
        return exponential_shifted(x,c,d,x0)

    params_exp, _ = curve_fit(
        exponential,
        x_exp,
        y_exp_residual,
        p0=[-0.01, 0.05],
        bounds = [[-0.5,0],[0,0.5]],    
        maxfev=5000             
    )
    c,d = params_exp

    full_fit = np.zeros_like(capacity)
    full_fit[:i] = linear(capacity[:i], a, b)
    full_fit[i:] = linear(capacity[i:], a, b) + exponential(capacity[i:], c, d)

    mse = mean_squared_error(voltage,full_fit)

    return mse

for filename in filelist:
    capacity = []
    voltage = []
    file_path = os.path.join(data_dir,filename)
    with open(file_path,'r') as f:
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

    mse_values = []
    cutoff_percent = []

    voltage = savgol_filter(voltage, window_length=151, polyorder=3)
    flatten_index = int(0.07 * len(capacity))
    capacity = np.array(capacity)
    voltage = np.array(voltage)

    capacity = capacity[flatten_index:]
    voltage = voltage[flatten_index:]
    capacity = capacity[::10]
    voltage = voltage[::10]
    capacity_len = len(capacity)


    for i in range(5,95,5):
        cutoff_index = int(capacity_len/100 * i)
        try:   
            mse = eval_model(capacity,voltage,filename,cutoff_index)
            mse_values.append(mse)
            cutoff_percent.append(i)
            print(mse)
        except Exception as e:
            print(f"{filename} failed at {i}% cutoff: {e}" )
        
        
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(capacity, voltage, label=filename)
    axs[0].legend()
    axs[0].set_title("Capacity vs Voltage")
    axs[0].set_xlabel("Capacity (mAh)")
    axs[0].set_ylabel("Voltage (V)")
    axs[0].grid(True)

    axs[1].plot(cutoff_percent, mse_values, marker='o')
    axs[1].set_title("MSE vs Cutoff %")
    axs[1].set_xlabel("Cutoff Percentage (%)")
    axs[1].set_ylabel("Mean Squared Error")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()



    