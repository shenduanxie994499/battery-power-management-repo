#this files stitches all of the csv files together and puts them into a folder called processed_data

import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from collections import *
from multivar import * 
from scipy.signal import savgol_filter

file_list = [f for f in os.listdir("./rawdata_singlepulse") if f.endswith('.csv')]
file_groups = group_files(file_list)

for key, group in file_groups.items():
    I1, T1, I2, T2, _, _ = extract_parameters(group[0][1])
    DC = T1/(T1 + T2)
    I_avg = average_current(I1, T1, I2, T2)
    T = T1 + T2

    time_offset = 0
    time = []
    voltage = []
    data_started = False
    for start, name in group:
        with open(os.path.join("./rawdata_singlepulse", name),'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if not any(row):
                    continue  # skip empty rows

                # Extract sample interval
                if "Sample interval:" in row[0]:
                    try:
                        sample_interval = float(row[0].split(":")[1].strip())
                    except:
                        print("Sample interval not retrieved")
                        pass

                # Detect start of data
                if row[0].startswith("Enable:"):
                    data_started = True
                    continue 

                # Parse actual data rows
                if data_started:
                    try:
                        t = float(row[0]) * sample_interval + time_offset
                        v = float(row[1])
                        time.append(t / 3600)
                        voltage.append(v)
                    except:
                        continue

        _,_,_,_,start,end = extract_parameters(name)
        time_offset = (end - start) * 3600 + time_offset #offset the timescale for the next sequence of data

    time_arr = np.array(time)
    voltage_arr = np.array(voltage)
    capacity_arr = I_avg * time_arr
    #voltage_arr = savgol_filter(voltage_arr, window_length=11, polyorder=3)

    output_path = os.path.join("/Users/tomhuang/Documents/battery-power-management-repo/processed_data", f"{key}.csv")
    with open(output_path, "w", newline = "") as f:
        writer = csv.writer(f)
        writer.writerow(["Capacity_mAh", "Voltage_V"])
        writer.writerows(zip(capacity_arr, voltage_arr))



    

