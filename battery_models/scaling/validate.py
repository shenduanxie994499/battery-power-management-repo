import csv
import os
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import *
import pandas as pd
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import csv

# Load reference curve
ref_df = pd.read_csv("reference_curve.csv")
ref_sod = ref_df["sod"].values
ref_voltage = ref_df["fit"].values

# Load scale factors
scale_df = pd.read_csv("scalefactors.csv")

# Set data folder path
data_dir = "./processed_data_sod"

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
    
# function to calculate average discharge current based on discharge waveform
def average_current(on_current, on_time, off_current, off_time):
    return (on_current * on_time + off_current * off_time) / (on_time + off_time)


# Plot each scaled reference curve against experimental data
for idx, row in scale_df.iterrows():
    fname = row["filename"]
    scale_factor = row["scale_factor"]

    # Load experimental data
    filepath = os.path.join(data_dir, fname)
    voltage = []
    sod = []

    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row_data in reader:
            if not row_data or len(row_data) < 2:
                continue
            try:
                sod.append(float(row_data[0]))
                voltage.append(float(row_data[1]))
            except:
                continue

    sod = np.array(sod)
    voltage = np.array(voltage)

    # Interpolate scaled reference voltage
    scaled_sod = ref_sod / scale_factor
    interp_func = interp1d(scaled_sod, ref_voltage, bounds_error=False, fill_value="extrapolate")
    predicted_voltage = interp_func(sod)


    I1,T1,I2,T2,_,_ = extract_parameters(fname)
    Iavg = average_current(I1,T1,I2,T2)

    # Plot comparison
    plt.figure(figsize=(8, 5))
    plt.plot(sod, voltage, alpha = 0.5, label="Experimental", linewidth=2)
    plt.plot(sod, predicted_voltage, label="Scaled Reference", linewidth=2)
    plt.xlabel("SOD")
    plt.ylabel("Voltage (V)")
    plt.title(f"{fname} — {Iavg}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
