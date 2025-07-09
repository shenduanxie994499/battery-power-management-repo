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

# function to calculate average discharge current based on discharge waveform
def average_current(on_current, on_time, off_current, off_time):
    return (on_current * on_time + off_current * off_time) / (on_time + off_time)

#fit a polynomial fit to the cropped voltage discharge curve and write it into a dictionary
def fit_model(capacity, voltage, filename, polydeg = 7, fev = 2.2):
    I1, T1, I2, T2, _, _ = extract_parameters(filename)
    DC = T1/(T1 + T2)
    T = T1 + T2

    capacity = np.array(capacity)
    voltage = np.array(voltage)
    
    # Fit polynomial and compute derivative
    coeffs = np.polyfit(capacity, voltage, deg=5)
    poly = np.poly1d(coeffs)
    dpoly = poly.deriv()
    dV = dpoly(capacity)

    voltage = savgol_filter(voltage, window_length=15501, polyorder=3)
    
    flatten_index = int(0.07 * len(capacity))
    capacity = capacity[flatten_index:]
    voltage = voltage[flatten_index:]
    dV = dV[flatten_index:]

    plt.plot(capacity,voltage, label = filename)


    #poly_trimmed = np.poly1d(coeffs_trimmed)
    #voltage_fit = poly_trimmed(capacity_trimmed)

    #return capacity_trimmed, voltage_trimmed, voltage_fit, capacity, voltage

    # return {
    #     "DC" : DC,
    #     "I (mA)" : I1,
    #     "T (ms)" : T,
    #     **{f"coef_{i}" : c for i,c in enumerate(coeffs_trimmed[::-1])}
    # }



plt.figure()
for file in filelist:
    file_path = os.path.join(data_dir, file)

    # Load data
    capacity = []
    voltage = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if not row or len(row) < 2:
                continue
            try:
                c = float(row[0])  # capacity in mAh
                v = float(row[1])
                capacity.append(c)
                voltage.append(v)
            except:
                continue

    fit_model(capacity,voltage,file)

plt.xlabel("Capacity (mAh)")
plt.ylabel("Voltage (V)")
plt.title("Voltage vs Capacity (Predicted Curve)")
# plt.xlim(0,15)
# plt.ylim(0.9,1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# df = pd.DataFrame(results)
# X = df[["DC", "I (mA)", "T (ms)"]]
# Y = df[[col for col in df.columns if col.startswith("coef_")]] 

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# model = PLSRegression(n_components=2)
# model.fit(X_train, Y_train)

# new_input = [[0.4, 5, 5]]  # example: DC = 0.4, I = 30 mA, T = 10 ms
# predicted_coeffs = model.predict(new_input)[0]
# poly_pred = np.poly1d(predicted_coeffs[::-1])

# capacity_range = np.linspace(0, 50, 200)
# voltage_pred = poly_pred(capacity_range)

# # Plot the predicted curve
# plt.figure(figsize=(8, 4))
# plt.plot(capacity_range, voltage_pred, label='Predicted Voltage Curve')
# plt.xlabel("Capacity (mAh)")
# plt.ylabel("Voltage (V)")
# plt.title("Voltage vs Capacity (Predicted Curve)")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

# score = model.score(X_test, Y_test)
# print(f"R² score on test set: {score:.3f}")

