import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "processed_data")
filelist = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

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
def fit_model(capacity, voltage, filename, polydeg = 3, fev = 2.2):
    I1, T1, I2, T2, _, _ = extract_parameters(filename)
    DC = T1/(T1 + T2)
    T = float(T1 + T2)
    I_avg = average_current(I1, T1, I2, T2)

    capacity = np.array(capacity)
    voltage = np.array(voltage)
    
    mask = (voltage > fev) 
    capacity = capacity[mask]
    voltage = voltage[mask] 
    #voltage = savgol_filter(voltage, window_length=21, polyorder=3)

    # Fit polynomial and compute derivative
    # coeffs = np.polyfit(capacity, voltage, deg=5)
    # poly = np.poly1d(coeffs)
    # dpoly = poly.deriv()
    # dV = dpoly(capacity)

    # Flattening detection
    # epsilon = 0.005
    # N = 500
    # flatten_index = None
    # for i in range(skip, len(dV) - N):
    #     if np.all(np.abs(dV[i:i + N]) < epsilon):
    #         flatten_index = i
    #         break
    flatten_index = int(0.07 * len(capacity))
    capacity_trimmed = capacity[flatten_index:]
    voltage_trimmed = voltage[flatten_index:]

    coeffs_trimmed = np.polyfit(capacity_trimmed, voltage_trimmed, deg=polydeg)
    poly_trimmed = np.poly1d(coeffs_trimmed)

    # Solve: poly_trimmed(Q) = 2.4  →  poly_trimmed(Q) - 2.4 = 0
    target_voltage = 2.4
    roots = (poly_trimmed - target_voltage).roots

    # Filter for real roots where capacity > 0
    valid_roots = [r.real for r in roots if np.isreal(r) and r > 0]

    # Sort roots just to be safe, and pick the smallest positive one
    cutoff_capacity = min(valid_roots) if valid_roots else None

    return {
        "DC" : DC,
        "I (mA)" : I1,
        "T (ms)" : T,
        "cutoff_capacity": cutoff_capacity
    }

results = []
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

    results.append(fit_model(capacity,voltage,file))

df = pd.DataFrame(results)
print(df)
df.to_excel('output.xlsx')

X = df[["DC", "I (mA)", "T (ms)"]]
Y = df["cutoff_capacity"] 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, Y_train)

# new_input = [[0.4, 30, 5]]  # example: DC = 0.4, I = 30 mA, T = 10 ms
# prediction = model.predict(new_input)

score = model.score(X_test, Y_test)
print(f"R² score on test set: {score:.3f}")

Y_pred = model.predict(X_test)

for col in ["DC", "I (mA)", "T (ms)"]:
    plt.scatter(df[col], df["cutoff_capacity"])
    plt.xlabel(col)
    plt.ylabel("cutoff_capacity")
    plt.title(f"{col} vs cutoff_capacity")
    plt.grid(True)
    plt.show()
