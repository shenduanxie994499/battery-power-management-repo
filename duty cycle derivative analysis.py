import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Define the fitting function
def voltage_model(capacity, a, b, c, d, e):
    return a * capacity + b + c * np.exp(d * capacity + e)

def derivative_model(capacity, a, b, c, d, e):
    return a + c * d * np.exp(d * capacity + e)

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

# Tipping Point Calculation and Plotting
tipping_point_list = []
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
        derivative = derivative_model(capacity, *popt)
        print(f'Duty Cycle {int(dc * 100)}%:')  # printing the fitting parameters
        print(f'  a={popt[0]:.6e}, b={popt[1]:.6e}, c={popt[2]:.6e}, d={popt[3]:.6e}, e={popt[4]:.6e}')
    except RuntimeError:
        print(f"Fit failed for {int(dc * 100)}% duty cycle")
        continue

    # Calculate the tipping point
    derivative_initial = derivative[0]
    derivative_end = derivative[-1]
    tipping_point_value = derivative_initial + 0.1 * (derivative_end - derivative_initial)

    # Find the capacity corresponding to the tipping point value
    tipping_point_index = np.argmax(derivative <= tipping_point_value)  # Find the first index where derivative <= tipping point value
    tipping_point_capacity = capacity[tipping_point_index]

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the original voltage curve
    ax1.plot(capacity, voltage, 'b.', label='Data')
    ax1.plot(capacity, fitted_voltage, 'r-', label='Fitted Curve')
    ax1.set_xlabel('Capacity (mAh)')
    ax1.set_ylabel('Voltage (V)')
    ax1.set_title(f'Binary-pulsed discharge with {int(dc*100)}% Duty Cycle')
    ax1.grid(True)

    # Create a second y-axis for the derivative curve
    ax2 = ax1.twinx()
    ax2.plot(capacity, derivative, 'g-', label='Derivative of the Curve')
    ax2.set_ylabel('Derivative of Voltage (V/mAh)')

    # Mark the tipping point on the plot
    ax2.plot(tipping_point_capacity, derivative[tipping_point_index], 'ro', label='Tipping Point')

    # Add legends for both y-axes
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    plt.tight_layout()

    #record the tipping point
    tipping_point_list.append(tipping_point_capacity)
plt.show()
#print and plot the tipping point
tipping_point_coeff = np.polyfit([dc for i, dc in enumerate(duty_cycles) if i != 3], [tp for i, tp in enumerate(tipping_point_list) if i !=3], 2)  # Exclude 40% duty cycle data (index 3)
poly_func = np.poly1d(tipping_point_coeff)
x_fit = np.linspace(0.1, 0.5, 100)
y_fit = poly_func(x_fit)

for index, value in enumerate(tipping_point_list):
    print(f"the tipping point of the curve when duty cycle is {index*10} % is {value}")

plt.plot(duty_cycles,tipping_point_list,'b.',label = 'tipping points')
plt.plot(x_fit,y_fit,'r-',label = 'fitted tipping points')
plt.plot()
plt.xlabel('duty cycle')
plt.ylabel('tipping point capacity (mAh)')
plt.show()
