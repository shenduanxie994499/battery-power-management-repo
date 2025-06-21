import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Model function with exponential term
def discharge_model(x, a, b, c, d, e):
    return a * x + b + c * np.exp(d * x + e)

# CSV file list
csv_files = [
    '20mA1msec-0.2mA4msec0-24hour.csv',  '20mA1msec-0.2mA4msec24-48hour.csv',
    '20mA2msec-0.2mA8msec0-24hour.csv',  '20mA2msec-0.2mA8msec24-48hour.csv',
    '20mA3msec-0.2mA12msec0-24hour.csv', '20mA3msec-0.2mA12msec24-48hour.csv',
    '20mA4msec-0.2mA16msec0-24hour.csv', '20mA4msec-0.2mA16msec24-48hour.csv',
    '20mA5msec-0.2mA20msec0-24hour.csv', '20mA5msec-0.2mA20msec24-48hour.csv',
    '20mA6msec-0.2mA24msec0-24hour.csv', '20mA6msec-0.2mA24msec24-48hour.csv',
    '20mA7msec-0.2mA28msec0-24hour.csv', '20mA7msec-0.2mA28msec24-48hour.csv',
    '20mA8msec-0.2mA32msec0-24hour.csv', '20mA8msec-0.2mA32msec24-48hour.csv',
]

# 8 unique initial guesses for curve fitting
initial_guesses = [
    [-2.900370e-04, 2.744200, -4.902816e-05, 2.345335e-02, 5.815633],
    [-2.985618e-03, 2.754787, -9.617648e-08, 9.115211e-02, 3.754636],
    [-1.867720e-03, 2.754787, -1.700479e-05, 5.330070e-02, 3.435200],
    [-8.303640e-04, 2.754787, -1.561092e-05, 3.849289e-02, 5.910279],
    [-1.171510e-03, 2.754787, -1.609146e-05, 3.849289e-02, 5.310279],
    [-7.247480e-04, 2.754787, -2.261090e-05, 3.849289e-02, 5.610279],
    [-8.857543e-04, 2.754787, -2.661861e-05, 3.149289e-02, 5.810279],
    [-5.037758e-04, 2.760638, -2.819467e-05, 2.983179e-02, 6.211609],
]

# CSV reading function
def read_csv_filtered(file_path):
    time_data, voltage_data = [], []
    if os.path.isfile(file_path):
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                try:
                    if len(row) >= 2:
                        time = float(row[0])
                        voltage = float(row[1])
                        time_data.append(time)
                        voltage_data.append(voltage)
                except ValueError:
                    continue
    return time_data, voltage_data

# Store fitted parameters
fitted_parameters = []
all_c_filtered = []
all_v_filtered = []
all_labels = []

# Process each pair
for i in range(0, len(csv_files), 2):
    curve_idx = i // 2  # 0 to 7
    t1, v1 = read_csv_filtered(csv_files[i])
    t2, v2 = read_csv_filtered(csv_files[i + 1])

    if t1 and t2:
        offset = t1[-1]
        t2 = [t + offset for t in t2]

        full_time = t1 + t2
        full_voltage = v1 + v2

        # Downsample and filter
        t_sampled = full_time[::100]
        v_sampled = full_voltage[::100]

        t_filtered = [t for t, v in zip(t_sampled, v_sampled) if 2.2 <= v <= 2.8]
        v_filtered = [v for v in v_sampled if 2.2 <= v <= 2.8]
        c_filtered = [0.5 * 4.16 * c / 3600 for c in t_filtered]  # Convert to capacity (mAh)

        # Use the specific initial guess for this curve
        p0 = initial_guesses[curve_idx]

        all_c_filtered.append(c_filtered)
        all_v_filtered.append(v_filtered)
        all_labels.append(f'{(curve_idx + 1) * 5} ms')




        try:
            popt, _ = curve_fit(discharge_model, c_filtered, v_filtered, p0=p0)
            fitted_parameters.append(popt)

            # Plot each curve in a new figure
            plt.figure(figsize=(8, 5))
            plt.scatter(c_filtered, v_filtered, s=10, label='Sampled Data', color='blue')
            x_fit = np.linspace(min(c_filtered), max(c_filtered), 500)
            y_fit = discharge_model(x_fit, *popt)
            plt.plot(x_fit, y_fit, label='Fitted Curve', color='red')

            plt.xlabel('Capacity (mAh)')
            plt.ylabel('Voltage (V)')
            plt.title(f'{(curve_idx + 1) * 5} ms binary pulsed period')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

        except RuntimeError:
            print(f"Curve fitting failed for pair {curve_idx + 1}")

# Plot all sampled curves before fitting in one figure
plt.figure(figsize=(10, 6))
for c_data, v_data, label in zip(all_c_filtered, all_v_filtered, all_labels):
    plt.scatter(c_data, v_data, s=10, label=label)

plt.xlabel('Capacity (mAh)')
plt.ylabel('Voltage (V)')
plt.title('Binary pulsed discharge curve with different pulse periods')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
           

# Print fitted parameters, and store the parameters in arrays
para_a, para_b, para_c, para_d, para_e = [],[],[],[],[]
for idx, ( a, b, c, d, e) in enumerate(fitted_parameters, start=1):
    print(f'Pair {idx}: a = {a:.6e}, b = {b:.6e}, c = {c:.6e}, d = {d:.6e}, e = {e:.6e}')
    para_a.append(a)
    para_b.append(b)
    para_c.append(c)
    para_d.append(d)
    para_e.append(e)

#poly fitting of parameters
# Polynomial fitting of parameters
pulse_period = [5,10,15,20,25,30,35,40]  # Pulse period array in msec
poly_order = 3
parameter_names = ['a', 'b', 'c', 'd', 'e']
parameter_lists = [para_a, para_b, para_c, para_d, para_e]
coefficient_combination = []

for i, (name, param_values) in enumerate(zip(parameter_names, parameter_lists)):
    coeffs = np.polyfit(pulse_period, param_values, poly_order)
    coefficient_combination.append(coeffs)

    # Print polynomial expression
    terms = [f"({c:.6e}) * T^{poly_order - j}" for j, c in enumerate(coeffs)]
    expression = f"{name}(T) = " + " + ".join(terms)
    print(expression)

    # Plotting in separate figures
    x_fit = np.linspace(min(pulse_period), max(pulse_period), 200)
    y_fit = np.polyval(coeffs, x_fit)

    plt.figure(figsize=(8, 5))
    plt.plot(x_fit, y_fit, label=f'{name}(T)', color='red')
    plt.scatter(pulse_period, param_values, label=f'{name} data', color='blue')
    plt.xlabel('Pulse Period T (ms)')
    plt.ylabel(f'Parameter {name}')
    plt.title(f'Polynomial Fit for Parameter {name}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
