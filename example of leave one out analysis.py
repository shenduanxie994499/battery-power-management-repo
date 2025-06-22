import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# List of CSV files
csv_files = ['0.5mA continious.csv', '1mA continious.csv', '1.5mA continious.csv',
             '2mA continious.csv', '2.5mA continious.csv', '3mA continious.csv']
drain_currents = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

def model(x, a, b, c, d, e):
    return a * x + b + c * np.exp(d * x + e)

initial_guesses = [
    [-0.00002, 2.8, -0, 0.1, 0],
    [-0.00003, 2.8, -0, 0.1, 0],
    [-0.00004, 2.8, -0, 0.1, 0],
    [-0.00005, 2.8, -0, 0.1, 0],
    [-0.00008, 2.8, -0, 0.1, 0],
    [-0.00008, 2.8, -0, 0.1, 0]
]

# Parameter storage
a_vals, b_vals, c_vals, d_vals, e_vals = [], [], [], [], []

# Step 1: Extract parameters from all CSVs
for index, csv_file in enumerate(csv_files):
    column1 = []
    column2 = []

    if os.path.isfile(csv_file):
        with open(csv_file, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                try:
                    x = float(row[0])
                    y = float(row[1])
                    column1.append(x)
                    column2.append(y)
                except (ValueError, IndexError):
                    continue

        if len(column1) == 0:
            print(f"No valid data in {csv_file}")
            continue

        x_data = np.array([x for x, y in zip(column1, column2) if y >= 2.2])
        y_data = np.array([y for x, y in zip(column1, column2) if y >= 2.2])

        try:
            p0 = initial_guesses[index]
            popt, _ = curve_fit(model, x_data, y_data, p0=p0, maxfev=10000)
            a_vals.append(popt[0])
            b_vals.append(popt[1])
            c_vals.append(popt[2])
            d_vals.append(popt[3])
            e_vals.append(popt[4])
        except RuntimeError:
            print(f"Could not fit data from {csv_file}")
            a_vals.append(np.nan)
            b_vals.append(np.nan)
            c_vals.append(np.nan)
            d_vals.append(np.nan)
            e_vals.append(np.nan)

# Step 2: Polynomial fitting and error analysis
param_names = ['a', 'b', 'c', 'd', 'e']
param_values = [a_vals, b_vals, c_vals, d_vals, e_vals]
fig, axs = plt.subplots(3, 2, figsize=(14, 12))
axs = axs.flatten()

poly_degree = 2

for i, (param_name, values) in enumerate(zip(param_names, param_values)):
    values = np.array(values)
    
    # Full polynomial fit
    full_coeffs = np.polyfit(drain_currents, values, poly_degree)
    full_fit = np.polyval(full_coeffs, drain_currents)
    full_residuals = values - full_fit
    full_error = np.sum(full_residuals**2)
    
    # Leave-one-out analysis
    max_error = -np.inf
    worst_idx = -1
    for j in range(len(drain_currents)):
        temp_currents = np.delete(drain_currents, j)
        temp_values = np.delete(values, j)

        try:
            temp_coeffs = np.polyfit(temp_currents, temp_values, poly_degree)
            temp_fit = np.polyval(temp_coeffs, values=drain_currents)
            temp_residuals = values - temp_fit
            temp_error = np.sum(temp_residuals**2)
            if temp_error > max_error:
                max_error = temp_error
                worst_idx = j
        except:
            continue

    # Exclude the worst point and refit
    filtered_currents = np.delete(drain_currents, worst_idx)
    filtered_values = np.delete(values, worst_idx)
    improved_coeffs = np.polyfit(filtered_currents, filtered_values, poly_degree)
    improved_fit = np.polyval(improved_coeffs, drain_currents)
    improved_residuals = values - improved_fit
    improved_error = np.sum(improved_residuals**2)

    # Plot data with error bars (absolute residuals as errors)
    axs[i].errorbar(drain_currents, values, yerr=np.abs(full_residuals),
                    fmt='o', label='Original Data', color='blue', capsize=4)

    # Full poly fit
    axs[i].plot(drain_currents, full_fit, '-', label=f'Poly Fit (all)\nErr={full_error:.2e}', color='green')

    # Improved poly fit (excluding worst)
    axs[i].plot(drain_currents, improved_fit, '--', label=f'Poly Fit (excl. {drain_currents[worst_idx]:.1f} mA)\nErr={improved_error:.2e}', color='orange')

    axs[i].set_title(f"Parameter {param_name} vs Drain Current")
    axs[i].set_xlabel("Drain Current (mA)")
    axs[i].set_ylabel(f"{param_name}")
    axs[i].legend()
    axs[i].grid(True)

# Hide the unused subplot
axs[-1].axis('off')

plt.tight_layout()
plt.show()
