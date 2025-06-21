import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Define the fitting function
def voltage_model(capacity, a, b, c, d, e):
    return a * capacity + b + c * np.exp(d * capacity + e)

csv_files = [
    '30mA1msec-0.2mA9msec0-24hour',
    '30mA1msec-0.2mA9msec24-48hour',
    '30mA2msec-0.2mA8msec0-24hour',
    '30mA3msec-0.2mA7msec0-24hour',
    '30mA4msec-0.2mA6msec0-24hour',
    '30mA5msec-0.2mA5msec0-24hour'
]

duty_cycles = [0.1, 0.2, 0.3, 0.4, 0.5]

data_sets = []

combined_time = []
combined_voltage = []

for i, file in enumerate(csv_files):
    time = []
    voltage = []

    with open(file + '.csv', 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            try:
                t = float(row[0])
                v = float(row[1])
                time.append(t)
                voltage.append(v)
            except ValueError:
                continue

    time = [x / 7200 for x in time]  # convert to hours, sampling every 0.5sec
    time = np.array(time)
    voltage = np.array(voltage)

    # Filter voltage range
    mask = (voltage >= 2.2) & (voltage <= 2.8)
    time = time[mask]
    voltage = voltage[mask]

    if i == 0:
        combined_time = time
        combined_voltage = voltage
    elif i == 1:
        time_offset = combined_time[-1]
        time = time + time_offset
        combined_time = np.concatenate((combined_time, time))
        combined_voltage = np.concatenate((combined_voltage, voltage))
        data_sets.append((combined_time, combined_voltage))
    else:
        data_sets.append((time, voltage))

# Curve fitting per dataset with individual initial guesses
initial_guesses = [
    [-4.361987e-04, 2.766759e+00, -4.494255e-08, 3.376468e-02, 1.271104e+01],  # for 10% duty
    [-2.635938e-03, 2.653940e+00, -5.018709e-08, 9.135696e-02, 1.090703e+01],  # for 20%
    [-1.352700e-03, 2.574400e+00, -4.928452e-09, 1.160005e-01, 1.431877e+01],  # for 30%
    [-1.093986e-02, 2.514378e+00, -9.459073e-12, 4.042848e-01, 1.503630e+01],   # for 40%;  this data create large error
    [-1.270010e-02, 2.487272e+00, -1.810338e-09, 5.143958e-01, 1.049961e+01]    # for 50%
]

# === Plot raw (unfiltered) time Capacity-Voltage curves ===

plt.figure()
colors = ['b', 'g', 'r', 'c', 'm']
for i, (time, voltage) in enumerate(data_sets):
    dc = duty_cycles[i]

    plt.plot(time, voltage, '.', color=colors[i], label=f'{int(dc * 100)}% Duty Cycle')

plt.xlabel('time (h)')
plt.ylabel('Voltage (V)')
plt.title('Duty cycle test in time domain')
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.figure()
colors = ['b', 'g', 'r', 'c', 'm']
for i, (time, voltage) in enumerate(data_sets):
    dc = duty_cycles[i]
    current = 20 * dc + 0.2 * (1 - dc)  # mA
    capacity = time * current  # mAh

    plt.plot(capacity, voltage, '.', color=colors[i], label=f'{int(dc * 100)}% Duty Cycle')

plt.xlabel('Capacity (mAh)')
plt.ylabel('Voltage (V)')
plt.title('Duty cycle test in capacity domain')
plt.grid(True)
plt.legend()
plt.tight_layout()


for i, (time, voltage) in enumerate(data_sets):
    dc = duty_cycles[i]
    current = 20 * dc + 0.2 * (1 - dc)  # mA
    capacity = time * current  # mAh

    # Filter by capacity ≥ 5 mAh
    mask = capacity >= 2.5
    capacity = capacity[mask]
    voltage = voltage[mask]

    initial_guess = initial_guesses[i]

    try:
        popt, _ = curve_fit(voltage_model, capacity, voltage, p0=initial_guess, maxfev=10000)
        fitted_voltage = voltage_model(capacity, *popt)
        print(f'Duty Cycle {int(dc * 100)}%:')
        print(f'  a={popt[0]:.6e}, b={popt[1]:.6e}, c={popt[2]:.6e}, d={popt[3]:.6e}, e={popt[4]:.6e}')
    except RuntimeError:
        print(f"Fit failed for {int(dc * 100)}% duty cycle")
        continue

    # Plot
    plt.figure()
    plt.plot(capacity, voltage, 'b.', label='Data')
    plt.plot(capacity, fitted_voltage, 'r-', label='Fitted Curve')
    plt.xlabel('Capacity (mAh)')
    plt.ylabel('Voltage (V)')
    plt.title(f'Binary-pulsed discharge with {int(dc*100)}% Duty Cycle')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

# === Fit polynomials to parameters vs. duty cycle ===

# Collect optimized parameters for each duty cycle
params = {'a': [], 'b': [], 'c': [], 'd': [], 'e': []}
used_duty_cycles = []

for i, (time, voltage) in enumerate(data_sets):
    dc = duty_cycles[i]
    current = 20 * dc + 0.2 * (1 - dc)
    capacity = time * current
    mask = capacity >= 2.5
    capacity = capacity[mask]
    voltage = voltage[mask]

    initial_guess = initial_guesses[i]
    try:
        popt, _ = curve_fit(voltage_model, capacity, voltage, p0=initial_guess, maxfev=10000)
        params['a'].append(popt[0])
        params['b'].append(popt[1])
        params['c'].append(popt[2])
        params['d'].append(popt[3])
        params['e'].append(popt[4])
        used_duty_cycles.append(dc)
    except RuntimeError:
        print(f"Skipping polynomial fit for {int(dc*100)}% due to fitting failure")


# Fit and plot each parameter separately
param_names = ['a', 'b', 'c', 'd', 'e']

for name in param_names:
    y = params[name]
    x = used_duty_cycles

    coeffs = np.polyfit(x, y, deg=3)
    poly_func = np.poly1d(coeffs)

    # Print the expression
    print(f"\nFitted polynomial for parameter '{name}':")
    print(f"{name}(dc) = ({coeffs[0]:.6e}) * dc^3 + ({coeffs[1]:.6e}) * dc^2 + ({coeffs[2]:.6e})* dc^1 + ({coeffs[3]:.6e})")

    # Plotting
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = poly_func(x_fit)

    plt.figure()
    plt.plot(x, y, 'o', label='Fitted Params')
    plt.plot(x_fit, y_fit, '-', label='Poly Fit (deg=3)')
    plt.xlabel('Duty Cycle')
    plt.ylabel(name)
    plt.title(f'{name} vs Duty Cycle')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()




plt.show()




#these parameters are incorrect
# initial_guesses = [
#     [-2.181000e-04, 2.766759, -9.821385e-90, 1.688235e-02, 2.007413e+02],  # for 10% duty
#     [-1.317969e-03, 2.653940, -9.071400e-45, 4.567848e-02, 9.551072e+01],  # for 20%
#     [-6.763498e-04, 2.574400, -7.593149e-28, 5.800024e-02, 5.763566e+01],  # for 30%
#     [-5.469932e-03, 2.514378, -1.270648e-19, 2.021425e-01, 3.223637e+01],  # for 40%
#     [-6.350079e-03, 2.487273, -2.537135e-13, 2.572009e-01, 1.937234e+01],  # for 50%
# ]

#these parameters are incorrect
# initial_guesses = [
#     [-2.180979e-04, 2.766759e+00, -6.357463e-09, 1.688233e-02, 1.466680e+01],  # for 10% duty
#     [-1.317974e-03, 2.653940e+00, -4.482266e-09, 4.567857e-02, 1.332265e+01],  # for 20%
#     [-6.763497e-04, 2.574400e+00, -4.113228e-09, 5.800024e-02, 1.449959e+01],  # for 30%
#     [-5.469932e-03, 2.514378e+00, -2.265451e-09, 2.021425e-01, 8.632279e+00],   # for 40%
#     [-6.350073e-03, 2.487273e+00, -1.061419e-08, 2.572003e-01, 8.730862e+00]    # for 50%
# ]

# initial_guesses = [
#     [-4.361987e-04, 2.766759e+00, -4.494255e-08, 3.376468e-02, 1.271104e+01],  # for 10% duty
#     [-2.635938e-03, 2.653940e+00, -5.018709e-08, 9.135696e-02, 1.090703e+01],  # for 20%
#     [-1.352700e-03, 2.574400e+00, -4.928452e-09, 1.160005e-01, 1.431877e+01],  # for 30%
#     [-1.093986e-02, 2.514378e+00, -9.459073e-12, 4.042848e-01, 1.503630e+01],   # for 40%;  this data create large error
#     [-1.270010e-02, 2.487272e+00, -1.810338e-09, 5.143958e-01, 1.049961e+01]    # for 50%
# ]