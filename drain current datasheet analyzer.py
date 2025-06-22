import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# List of CSV files and drain currents (in mA)
csv_files = ['0.5mA continious.csv', '1mA continious.csv', '1.5mA continious.csv',
             '2mA continious.csv', '2.5mA continious.csv', '3mA continious.csv']
plot_title = ['0.5mA','1.0mA','1.5mA','2.0mA','2.5mA','3.0mA']
drain_currents = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

# Define the model function: y = ax + b + c * exp(dx + e)
def model(x, a, b, c, d, e):
    return a * x + b + c * np.exp(d * x + e)

# Initial guesses for curve fitting
initial_guesses = [
    [-0.00002, 2.8, -0, 0.1, 0],
    [-0.00003, 2.8, -0, 0.1, 0],
    [-0.00004, 2.8, -0, 0.1, 0],
    [-0.00005, 2.8, -0, 0.1, 0],
    [-0.00008, 2.8, -0, 0.1, 0],
    [-0.00008, 2.8, -0, 0.1, 0]
]

# Store fitted parameters
fitted_parameters = []

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

        # Filter voltage > 2.2V
        filter_col1 = [x for x,y in zip(column1,column2) if y >= 2.2]
        filter_col2 = [y for x,y in zip(column1,column2) if y >= 2.2]

        x_data = np.array(filter_col1)
        y_data = np.array(filter_col2)

        try:
            p0 = initial_guesses[index]
            popt, _ = curve_fit(model, x_data, y_data, p0=p0, maxfev=10000)
            fitted_parameters.append(popt)

            print(f"Fitted parameters for {csv_file}: a={popt[0]:.6f}, b={popt[1]:.4f}, c={popt[2]:.15f}, d={popt[3]:.4f}, e={popt[4]:.4f}")

            # Plot raw and fitted curves
            plt.figure(figsize=(10, 6))
            plt.plot(x_data, y_data, 'o', color='blue', label=f'Data: {plot_title[index]}')
            x_fit = np.linspace(min(x_data), max(x_data), 500)
            y_fit = model(x_fit, *popt)
            plt.plot(x_fit, y_fit, '-', color='red', label='Fitted Curve')
            plt.xlabel("Capacity (mAh)")
            plt.ylabel("Voltage (V)")
            plt.title(f"Discharge curve for {plot_title[index]} drain current")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        except RuntimeError:
            print(f"Could not fit data from {csv_file}")

# Polynomial fitting for each parameter vs. drain current
if fitted_parameters:
    fitted_parameters = np.array(fitted_parameters)  # shape: (6, 5)
    param_names = ['a', 'b', 'c', 'd', 'e']
    degree = 2  # You can change this to 1 for linear or higher for more complex fits

    for i in range(5):
        coeffs = np.polyfit(drain_currents, fitted_parameters[:, i], degree)
        poly = np.poly1d(coeffs)
        print(f"\nPolynomial expression for {param_names[i]}(I):")
        print(poly)

        # Plot the trend
        plt.figure()
        plt.plot(drain_currents, fitted_parameters[:, i], 'o', label='Fitted Parameters')
        x_line = np.linspace(min(drain_currents), max(drain_currents), 200)
        plt.plot(x_line, poly(x_line), '-', label=f'Poly Fit (deg {degree})')
        plt.title(f"{param_names[i]} vs Drain Current")
        plt.xlabel("Drain Current (mA)")
        plt.ylabel(f"{param_names[i]}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
