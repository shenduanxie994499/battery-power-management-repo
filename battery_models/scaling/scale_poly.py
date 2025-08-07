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
from scipy.spatial.distance import mahalanobis
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor

data_dir = "./processed_data"
file_list = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

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

# avgs = []
# files = []
# for f in file_list:
#     I1,T1,I2,T2,_,_ = extract_parameters(f)
#     I_avg = average_current(I1,T1,I2,T2)
#     files.append(f)
#     avgs.append(I_avg)

# res = {'Name': files, 'Average Current': avgs}
# df = pd.DataFrame(res)
# print(df)
# print(np.median(avgs))

reference = "20.0mA6msec-0.2mA24msec.csv"
ref_filepath = os.path.join(data_dir,reference)

capacity = []
voltage = []
with open(ref_filepath,'r') as f:
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


capacity = np.array(capacity)
voltage = np.array(voltage)

mask = voltage >= 2.5
capacity = capacity[mask]
voltage = voltage[mask]

# Normalize capacity → SOD
sod = capacity / np.max(capacity)

# Fit 5th-degree polynomial: V = f(SOD)
coeffs = np.polyfit(sod, voltage, deg=11)  # highest degree first

# Evaluate fit
sod_fit = np.linspace(0, 1, 200)
voltage_fit = np.polyval(coeffs, sod_fit)

# Plot to visualize
plt.figure()
plt.plot(sod, voltage, label='Data', marker='o', linestyle='None', alpha=0.6)
plt.plot(sod_fit, voltage_fit, label='5th-degree fit', linewidth=2)
plt.xlabel('State of Discharge (SOD)')
plt.ylabel('Voltage (V)')
plt.title('Reference Curve Polynomial Fit')
plt.legend()
plt.grid(True)
plt.show()
    
