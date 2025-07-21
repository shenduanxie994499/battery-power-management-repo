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

    cutoff_mah = 5  # or whatever threshold you want
    mask = capacity >= cutoff_mah
    capacity = capacity[mask]
    voltage = voltage[mask]

    # Find cutoff point
    i = find_cutoff(capacity, voltage)
    plt.scatter(capacity[i], voltage[i], color='red', s=20, zorder=5)

    x_lin = capacity[:i]
    y_lin = voltage[:i]
    x_exp = capacity[i:]
    y_exp = voltage[i:]

    def linear(x,a,b):
        return a * x + b

    params_lin, _ = curve_fit(linear,x_lin,y_lin)
    a,b = params_lin

    y_exp_residual = y_exp - linear(x_exp, a, b)
    x0 = x_exp[0]

    def exponential(x, c, d):
        return c * (np.exp(d * x) - np.exp(d * x0))

    params_exp, _ = curve_fit(
        exponential,
        x_exp,
        y_exp_residual,
        p0=[-0.01, 0.05],
        bounds = [[-0.5,0],[0,0.1]],  
        maxfev=5000                 
    )
    c,d = params_exp

    full_fit = np.zeros_like(capacity)
    full_fit[:i] = linear(capacity[:i], a, b)
    full_fit[i:] = linear(capacity[i:], a, b) + exponential(capacity[i:], c, d)

    plt.plot(capacity, voltage, alpha=0.5, label=f"Data ({filename})")
    plt.plot(capacity, full_fit, label=f"Fit ({filename})", linewidth=2)

    print(f"{filename} | split at {capacity[i]:.2f} | a={a:.4f}, b={b:.4f}, c={c:.4e}, d={d:.4f}")

    coeffs = [a,b,c,d]
    return {
        "DC" : DC,
        "I (mA)" : I1,
        "T (ms)" : T,
        **{f"coef_{chr(i + ord('a'))}" : c for i,c in enumerate(coeffs)},
        "x0" : x0
    }

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

df = pd.DataFrame(results)
print(df)

X = df[["DC", "I (mA)", "T (ms)"]]
Y = df[[col for col in df.columns if col.startswith("coef_")] + ["x0"]] 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# model = PLSRegression(n_components=2)
# model.fit(X_train, Y_train)
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, Y_train)

DC = 0.2
I = 30
T = 10

new_input = [[0.4, 5, 5]]  # example: DC = 0.4, I = 30 mA, T = 10 ms
predicted_coeffs = model.predict(new_input)[0]
a, b, c, d, x0 = predicted_coeffs
c = -abs(c)

capacity_range = np.linspace(0, 500, 200)
split_index = np.searchsorted(capacity_range, x0)

def linear_eval(x):
    return a * x + b

def exponential_eval(x):
    return c * (np.exp(d * x) - np.exp(d * x0))


voltage_pred = np.zeros_like(capacity_range)
voltage_pred[:split_index] = linear_eval(capacity_range[:split_index])
voltage_pred[split_index:] = linear_eval(capacity_range[split_index:]) + exponential_eval(capacity_range[split_index:])

print(f"DC: {DC}, I: {I}, T: {T} | split at {x0:.2f} | a={a:.4f}, b={b:.4f}, c={c:.4e}, d={d:.4f}")

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

score = model.score(X_test, Y_test)
print(f"R² score on test set: {score:.3f}")

# T_train = model.transform(X_train)

# # Mean and covariance in score space
# T_mean = np.mean(T_train, axis=0)
# T_cov = np.cov(T_train, rowvar=False)
# T_cov_inv = np.linalg.inv(T_cov)

# # Hotelling's T² for each training sample
# T2_scores = [mahalanobis(t, T_mean, T_cov_inv)**2 for t in T_train]
# threshold = stats.chi2.ppf(0.95, df=model.n_components)
# outlier_indices = np.where(np.array(T2_scores) > threshold)[0]


# X_train = X_train.reset_index(drop=True)
# outlier_params = X_train.loc[outlier_indices]


# print("Training outliers based on T²:")
# for _, row in outlier_params.iterrows():
#     print(f"DC={row['DC']:.2f}, I={row['I (mA)']:.1f}, T={row['T (ms)']}")

#     plt.figure(figsize=(10, 5))
# bars = plt.bar(range(len(T2_scores)), T2_scores, color='skyblue')
# plt.axhline(threshold, color='red', linestyle='--', label=f"T² threshold ({threshold:.2f})")


# for idx in outlier_indices:
#     bars[idx].set_color('orange')
#     label = f"DC={X_train.iloc[idx]['DC']:.2f}, I={X_train.iloc[idx]['I (mA)']:.0f}, T={X_train.iloc[idx]['T (ms)']:.0f}"
#     plt.text(idx, T2_scores[idx] + 0.5, label, rotation=90, ha='center', va='bottom', fontsize=7)

# plt.xlabel("Training Sample Index")
# plt.ylabel("Hotelling's T²")
# plt.title("Hotelling’s T² on Training Data")
# plt.legend()
# plt.tight_layout()
# plt.grid(True)
# plt.show()


# # Get X reconstruction using scores and X loadings
# X_hat = np.dot(T_train, model.x_loadings_.T)

# # Compute residuals
# X_residual = X_train.to_numpy() - X_hat

# Q_residuals = np.sum(X_residual**2, axis=1)

# plt.figure(figsize=(10, 5))
# bars = plt.bar(range(len(Q_residuals)), Q_residuals, color='lightgray')

# plt.xlabel("Training Sample Index")
# plt.ylabel("Q-residual (SPE)")
# plt.title("Q-Residuals on Training Data")
# plt.legend()
# plt.tight_layout()
# plt.grid(True)
# plt.show()
