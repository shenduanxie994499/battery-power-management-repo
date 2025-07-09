import numpy as np
import matplotlib.pyplot as plt
import csv
from multivar_threepart import *

def linear_model(Q,a,b):
    return a * Q + b

def piecewise_model(capacity, voltage, polydeg = 5, slope_threshold = 0.1, dfrac= 0.1):
    coeffs = np.polyfit(capacity, voltage, deg=polydeg)
    poly = np.poly1d(coeffs)

    dpoly = poly.deriv()
    dV = dpoly(capacity)

    derivative_initial = dV[0]
    derivative_end = dV[-1]
    tipping_value = derivative_initial + dfrac * (derivative_end - derivative_initial)

    # Skip the first 10% of the data
    start_index = int(0.1 * len(capacity))
    tipping_index_relative = np.argmax(dV[start_index:] <= tipping_value)
    tipping_index = tipping_index_relative + start_index

    Q_lin = capacity[:tipping_index]
    V_lin = voltage[:tipping_index]
    Q_exp = capacity[tipping_index:]
    V_exp = voltage[tipping_index:]

    try:
            # Fit linear region: aQ + b_lin
            popt_lin, _ = curve_fit(linear_model, Q_lin, V_lin)
            a, b_lin = popt_lin

            # Residual after removing linear part from exponential region
            V_exp_residual = V_exp - (a * Q_exp + b_lin)

            # Fit exponential region with offset: c * exp(dQ + e) + b_exp
            def shifted_exp(Q, c, d, e, b_exp):
                return c * np.exp(d * Q + e) + b_exp

            popt_exp, _ = curve_fit(shifted_exp, Q_exp, V_exp_residual, maxfev=10000)
            c, d, e, b_exp = popt_exp

            # Total b includes both intercepts
            b = b_lin + b_exp
            return [a, b, c, d, e]
    except Exception as err:
        print(f"Fit failed: {err}")
        return None


# === CONFIG ===
csv_path = "/Users/tomhuang/Documents/battery-power-management-repo/testing/30mA5msec-0.2mA5msec0-24hour.csv"
I1,T1,I2,T2,_,_ = extract_parameters('30mA5msec-0.2mA5msec0-24hour.csv')
I_avg = average_current(I1,T1,I2,T2)

# === LOAD DATA ===
time = []
voltage = []
with open(csv_path, 'r') as f:
    reader = csv.reader(f)
    header_skipped = False
    for row in reader:
        if not any(row):
            continue
        if not header_skipped:
            header_skipped = True
            continue
        try:
            t = float(row[0])  # time in seconds
            v = float(row[1])
            time.append(t)
            voltage.append(v)
        except:
            continue

time = np.array(time)
voltage = np.array(voltage)

# === CONVERT TIME TO CAPACITY (mAh) ===
time_hr = time / 3600  # Convert to hours
capacity = I_avg * time_hr  # Capacity in mAh

# === FILTER VOLTAGE (optional) ===
mask = (voltage > 2.2) & (voltage < 2.8)
capacity = capacity[mask]
voltage = voltage[mask]

# === FIT THE MODEL ===
params = piecewise_model(capacity, voltage)
if params:
    a, b, c, d, e = params
    print(f"Fit Parameters:\na={a:.4e}, b={b:.4e}, c={c:.4e}, d={d:.4e}, e={e:.4e}")

    # Reconstruct the model curve
    fitted_voltage = a * capacity + b + c * np.exp(d * capacity + e)

    # === PLOT ===
    plt.plot(capacity, voltage, label='Data')
    plt.plot(capacity, fitted_voltage, label='Model Fit')
    plt.xlabel('Capacity (mAh)')
    plt.ylabel('Voltage (V)')
    plt.title('Piecewise Model Fit')
    plt.grid()
    plt.legend()
    plt.show()
else:
    print("Fit failed.")
