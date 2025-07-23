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
from sklearn.metrics import mean_squared_error

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

def find_cutoff(capacity, voltage, scan_start_frac=0.35, negative_run=10):
    d1 = np.gradient(voltage, capacity)
    d2 = np.gradient(d1, capacity)
    print(f"Max d2: {np.max(d2):.4e}, Min d2: {np.min(d2):.4e}")
    start_idx = int(scan_start_frac * len(capacity))
    for i in range(start_idx, len(d2) - negative_run):
        if np.all(d2[i:i + negative_run] < -0.000001):
            return i
    return -1  # fallback

#filelist = ['20.0mA1msec-0.2mA4msec.csv']
def capacity_to_time(capacity, voltage, target_voltage=0.8, min_cap=10):
    voltage = np.array(voltage)
    capacity = np.array(capacity)

    valid = capacity >= min_cap
    voltage = voltage[valid]
    capacity = capacity[valid]

    for i in range(1, len(voltage)):
        if voltage[i-1] > target_voltage and voltage[i] <= target_voltage:
            cap_cross = np.interp(target_voltage,
                                  [voltage[i-1], voltage[i]],
                                  [capacity[i-1], capacity[i]])
            return cap_cross

    return None


# function to calculate average discharge current based on discharge waveform
def average_current(on_current, on_time, off_current, off_time):
    return (on_current * on_time + off_current * off_time) / (on_time + off_time)


def exponential(x, a, b, c):
    return a * (np.exp(b * x)) + c

#fit a polynomial fit to the cropped voltage discharge curve and write it into a dictionary
def fit_model(capacity, voltage, filename,downsample_rate=10):
    I1, T1, I2, T2, _, _ = extract_parameters(filename)
    DC = T1 / (T1 + T2)
    T = T1 + T2
    
    capacity = capacity[::downsample_rate]
    voltage = voltage[::downsample_rate]

    capacity = np.array(capacity)
    voltage = np.array(voltage)


    flatten_index = int(0.07 * len(capacity))
    capacity = capacity[flatten_index:]
    voltage = voltage[flatten_index:]

    params_exp, _ = curve_fit(
        exponential,
        capacity,
        voltage,
        p0=[-0.01, 0.05, 0.98],
        bounds = [[-0.5,-0.1,0.9],[0,0.1,1]],  
        maxfev=5000             
    )
    a,b,c = params_exp

    full_fit = exponential(capacity,a,b,c)

    plt.plot(capacity, voltage, alpha=0.5, label=f"Data ({filename})")
    plt.plot(capacity, full_fit, label=f"Fit ({filename})", linewidth=2)

    print(f"{filename} | a={a:.4f}, b={b:.4f}, c={c:.4e}")

    coeffs = [a,b,c]
    return {
        "DC" : DC,
        "I (mA)" : I1,
        "T (ms)" : T,
        **{f"coef_{chr(i + ord('a'))}" : c for i,c in enumerate(coeffs)},
    }

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

df = pd.DataFrame(results)
print(df)


X = df[["DC", "I (mA)", "T (ms)"]]
Y = df[[col for col in df.columns if col.startswith("coef_")]] 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = PLSRegression(n_components=2)
model.fit(X_train, Y_train)

    
time_errors = []
actual_times = []
predicted_times = []
for file in filelist:
    file_path = os.path.join(data_dir, file)

    capacity = []
    voltage = []

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
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

    I1, T1, I2, T2, _, _= extract_parameters(file)
    DC = T1 / (T1 + T2)
    T = T1 + T2

    # === True cutoff time ===
    cap_actual_27 = capacity_to_time(capacity, voltage)
    
    if cap_actual_27 is None:
        print(f"{file}: Actual curve didn't reach 0.8")
        continue
    I_avg = average_current(I1, T1, I2, T2)
    t_actual = (cap_actual_27 / I_avg) 

    # === Predict coefficients ===
    input_features = pd.DataFrame([[DC, I1, T]], columns=["DC", "I (mA)", "T (ms)"])
    predicted_coeffs = model.predict(input_features)[0]
    a, b, c = predicted_coeffs

    # === Generate predicted curve ===
    capacity_arr = np.linspace(0, 500, 200)
    voltage_pred = exponential(capacity_arr,a,b,c)

    valid_mask = voltage_pred > 0
    capacity_arr = capacity_arr[valid_mask]
    voltage_pred = voltage_pred[valid_mask]

    plt.figure(figsize=(8, 4))
    plt.plot(capacity_arr, voltage_pred, label='Predicted', linewidth=2)
    plt.plot(capacity, voltage, label='Actual', linestyle='--', alpha=0.7)
    plt.title(f"{file} — Voltage vs Capacity")
    plt.xlabel("Capacity (mAh)")
    plt.ylabel("Voltage (V)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    cap_pred_27 = capacity_to_time(capacity_arr, voltage_pred)
    if cap_pred_27 is None:
        print(f"{file}: Predicted curve didn't reach 0.8")
        continue

    t_pred = (cap_pred_27 / I_avg) 
    actual_times.append(t_actual)
    predicted_times.append(t_pred)
    
    error = abs(t_actual - t_pred)
    time_errors.append(error)

    print(f"{file} | t_actual={t_actual:.2f}hr | t_pred={t_pred:.2f}hr | error={error:.2f}hr")

avg_error = np.mean(time_errors)
print(f"\nAverage absolute time error: {avg_error:.2f}hr")

mse = mean_squared_error(actual_times, predicted_times)
print(f"\nMean Squared Error: {mse:.4f} hr²")

rmse = np.sqrt(mse)
print(f"\nRoot Mean Squared Error: {rmse:.4f} hr")

