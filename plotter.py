#plotter.py visualizes all of the plots so we know which ones are good and which to throw out

import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from multivar import *

# === CONFIG ===
data_dir = "/Users/tomhuang/Documents/battery-power-management-repo/processed_data"
csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

# === PLOT MULTIPLE FILES ===
plt.figure()

for file_name in csv_files:
    file_path = os.path.join(data_dir, file_name)

    try:
        # Parse current info from filename
        I1, T1, I2, T2, _, _ = extract_parameters(file_name)
        I_avg = average_current(I1, T1, I2, T2)
    except:
        print(f"Skipping {file_name} — failed to extract parameters.")
        continue

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

    if capacity and voltage:
        capacity = np.array(capacity)
        voltage = np.array(voltage)
        plt.plot(capacity, voltage, label=file_name.replace(".csv", ""))

plt.xlabel('Capacity (mAh)')
plt.ylabel('Voltage (V)')
plt.title('Discharge Curves by Condition')
plt.grid(True)
plt.legend(fontsize="x-small", loc="best")
plt.tight_layout()
plt.show()
