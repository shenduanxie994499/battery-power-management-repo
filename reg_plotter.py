#reg_plotter.py visualizes all of the processed data plots so we know which ones are good and which to throw out

import csv
import os
import matplotlib.pyplot as plt
from multivar_threepart import *

# === CONFIG ===

# Get the directory where the script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Point to a 'processed_data' folder inside the same directory
data_dir = os.path.join(base_dir, "processed_data")
csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

# === PLOT MULTIPLE FILES ===
#plt.figure()

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

    cap_t, volt_t, fit,_,_ = model(capacity,voltage)
    plt.plot(cap_t,volt_t,label = "Original Data", marker = 'o', linestyle = "none")
    plt.plot(cap_t, fit, label = "polynomial fit", linestyle = '-')


    plt.xlabel('Capacity (mAh)')
    plt.ylabel('Voltage (V)')
    plt.title(f'{file_name} discharge curve')
    plt.grid(True)
    plt.legend(fontsize="x-small", loc="best")
    plt.tight_layout()
    plt.show()
