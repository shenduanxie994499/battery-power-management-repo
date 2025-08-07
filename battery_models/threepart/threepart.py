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

#fit a threepart fit to the cropped voltage discharge curve and write it into a dictionary
def fit_model(capacity, voltage, filename, downsample_rate=10):
    I1, T1, I2, T2, _, _ = extract_parameters(filename)
    DC = T1 / (T1 + T2)
    T = T1 + T2
    
    capacity = capacity[::downsample_rate]
    voltage = voltage[::downsample_rate]

    capacity = np.array(capacity)
    voltage = np.array(voltage)

    # Step 1: cutoff indices
    i = find_cutoff1(capacity, voltage)
    j = find_cutoff2(capacity, voltage)

    plt.scatter(capacity[i], voltage[i], color='red', s=20, zorder=5)
    plt.scatter(capacity[j], voltage[j], color='blue', s=20, zorder=5)

    # Step 2: fit exponential region 1 WITH offset
    x_exp1 = capacity[:i]
    y_exp1 = voltage[:i]

    def exp1(x, c, d, v0):
        return v0 + c * (np.exp(d * x) - 1)

    params1, _ = curve_fit(exp1, x_exp1, y_exp1, p0=[0.5, -12, 0.9])
    c1, d1, v0 = params1
    res1 = voltage - exp1(capacity, c1, d1, v0)

    # Step 3: fit linear region on residual
    x_lin = capacity[i:j]
    y_lin = res1[i:j]
    x1 = x_lin[0]

    def linear(x, a):
        return a * (x - x1)

    params_lin, _ = curve_fit(linear, x_lin, y_lin)
    a = params_lin[0]
    res2 = res1 - linear(capacity, a)

    # Step 4: fit second exponential region on second residual
    x_exp2 = capacity[j:]
    y_exp2 = res2[j:]
    x2 = x_exp2[0]

    def exp2(x, c, d):
        return c * (np.exp(d * (x - x2)) - 1)

    params2, _ = curve_fit(exp2, x_exp2, y_exp2, p0=[-0.1, 0.05])
    c2, d2 = params2

    # Preallocate final array
    full_fit = np.zeros_like(capacity)

    # Region 1: only exp1
    full_fit[:i] = exp1(capacity[:i], c1, d1, v0)

    # Region 2: exp1 + linear
    full_fit[i:j] = (
        exp1(capacity[i:j], c1, d1, v0) +
        linear(capacity[i:j], a)
    )

    # Region 3: exp1 + linear + exp2
    full_fit[j:] = (
        exp1(capacity[j:], c1, d1, v0) +
        linear(capacity[j:], a) +
        exp2(capacity[j:], c2, d2)
    )

    plt.plot(capacity, voltage, alpha=0.5, label=f"Data ({filename})")
    plt.plot(capacity, full_fit, label=f"Fit ({filename})", linewidth=2)

    return {
        "DC": DC,
        "I (mA)": I1,
        "T (ms)": T,
        "coef_c1": c1,
        "coef_d1": d1,
        "coef_v0": v0,
        "coef_a": a,
        "coef_c2": c2,
        "coef_d2": d2,
        "x1":x1,
        "x2":x2
    }

plt.figure()
results = []
for file in filelist:
    file_path = os.path.join(data_dir, file)

    capacity_range = []
    voltage_pred = []

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if not row or len(row) < 2:
                continue
            try:
                c = float(row[0])
                v = float(row[1])
                capacity_range.append(c)
                voltage_pred.append(v)
            except:
                continue

    result = fit_model(capacity_range, voltage_pred, file)
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

df = pd.DataFrame(results)
print(df)

X = df[["DC", "I (mA)", "T (ms)"]]
Y = df[[col for col in df.columns if col.startswith("coef_")] + ["x1"] + ["x2"]]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = PLSRegression(n_components=2)
model.fit(X_train, Y_train)

DC = 0.2
I = 30
T = 10

new_input = [[0.4, 5, 5]]  # example: DC = 0.4, I = 30 mA, T = 10 ms
predicted_coeffs = model.predict(new_input)[0]
c1,d1,v0,a,c2,d2,x1,x2 = predicted_coeffs
c2 = -abs(c2)

capacity_range = np.linspace(0, 500, 200)
voltage_pred = np.zeros_like(capacity_range)

# Define exp1
def exp1(x): return v0 + c1 * (np.exp(d1 * x) - 1)

# Define linear residual
def lin(x): return a * (x - x1)

# Define exp2
def exp2(x): return c2 * (np.exp(d2 * (x - x2)) - 1)

# Split into 3 regions
region1 = capacity_range < x1
region2 = (capacity_range >= x1) & (capacity_range < x2)
region3 = capacity_range >= x2

voltage_pred[region1] = exp1(capacity_range[region1])
voltage_pred[region2] = exp1(capacity_range[region2]) + lin(capacity_range[region2])
voltage_pred[region3] = exp1(capacity_range[region3]) + lin(capacity_range[region3]) + exp2(capacity_range[region3])

# voltage_pred = np.zeros_like(capacity_range)
# voltage_pred[:split_index] = linear_eval(capacity_range[:split_index])
# voltage_pred[split_index:] = linear_eval(capacity_range[split_index:]) + exponential_eval(capacity_range[split_index:])

print(
    f"DC: {DC}, I: {I}, T: {T} | "
    f"split at x1={x1:.2f}, x2={x2:.2f} | "
    f"c1={c1:.4f}, d1={d1:.4f}, v0={v0:.4f}, "
    f"a={a:.4f}, c2={c2:.4f}, d2={d2:.4f}"
)

valid_mask = voltage_pred > 0
capacity_range = capacity_range[valid_mask]
voltage_pred = voltage_pred[valid_mask]

# Plot the predicted curve
plt.figure()
plt.plot(capacity_range, voltage_pred, label='Predicted Voltage Curve')
plt.xlabel("Capacity (mAh)")

plt.ylabel("Voltage (V)")
plt.ylim(0,1)
plt.title("Voltage vs Capacity (Predicted Curve)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# score = model.score(X_test, Y_test)
# print(f"R² score on test set: {score:.3f}")