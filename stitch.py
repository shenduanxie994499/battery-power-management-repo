#this files stitches all of the csv files together and puts them into a folder
import re
import csv
import os
import numpy as np

from collections import *
# from multivar_threepart import * 
from scipy.signal import savgol_filter

# function to calculate average discharge current based on discharge waveform
def average_current(on_current, on_time, off_current, off_time):
    return (on_current * on_time + off_current * off_time) / (on_time + off_time)

# extracts parameters from filename
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

# Groups files that are continuations of one another (like 0-24 and 24-48 hour)
# Use a dictionary to do so, where the keys is the filename without the hours 
# and the value is a list of tuples in the form of (starthour, 'specific filename')
# The list of tuples is sorted in chronological order
def group_files(file_list):
    groups = defaultdict(list)

    for file in file_list:
        try:
            I1, T1, I2, T2, start, end = extract_parameters(file)
            key = f"{I1}mA{T1}msec-{I2}mA{T2}msec"
            groups[key].append((start,file))
        except ValueError:
            print("Matching failed for: " + file)
        
    for key in groups.keys():
        groups[key].sort(key=lambda x: x[0])

    return groups

file_list = [f for f in os.listdir("./rawdata_singlepulse") if f.endswith('.csv')]
file_groups = group_files(file_list)

for key, group in file_groups.items():
    I1, T1, I2, T2, _, _ = extract_parameters(group[0][1])
    DC = T1/(T1 + T2)
    T = T1 + T2
    I_avg = average_current(I1, T1, I2, T2)

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

#smooth data
files_list = [f for f in os.listdir("./processed_data") if f.endswith(".csv")]
fev = 2.2

for file in files_list:
    capacity = []
    voltage = []
    data_start = False
    with open(os.path.join("./processed_data",file),'r') as f:
        reader = csv.reader(f)
        for row in reader:
           if row[0].startswith("Capacity_mAh"):
               data_start = True
               continue
           
           if data_start:
               capacity.append(float(row[0]))
               voltage.append(float(row[1]))

    capacity = np.array(capacity)
    voltage = np.array(voltage)
    
    voltage = savgol_filter(voltage, window_length=1551, polyorder=3)

    mask = (voltage > fev) 
    capacity = capacity[mask]
    voltage = voltage[mask] 
    #window_size = 1501
    # kernel = np.ones(window_size) / window_size
    # voltage_smoothed = np.convolve(voltage, kernel, mode = 'valid')

    voltage = voltage / voltage[0]

    output_path = os.path.join("/Users/tomhuang/Documents/battery-power-management-repo/smoothed_normalized_data", file)
    with open(output_path, "w", newline = "") as f:
        writer = csv.writer(f)
        writer.writerow(["Capacity_mAh", "Voltage_V"])
        writer.writerows(zip(capacity, voltage))

# #this files stitches all of the csv files together and puts them into a folder
# import re
# import csv
# import os
# import numpy as np

# from collections import *
# # from multivar_threepart import * 
# from scipy.signal import savgol_filter

# # function to calculate average discharge current based on discharge waveform
# def average_current(on_current, on_time, off_current, off_time):
#     return (on_current * on_time + off_current * off_time) / (on_time + off_time)

# # extracts parameters from filename
# def extract_parameters(file_name):
#     """
#     Extract high_current, on_time, low_current, off_time, start_hour, end_hour from file name.
#     Example: '30mA1msec-0.2mA24msec0-24hour.csv' → 30.0, 1, 0.2, 24, 0, 24
#     """
#     match = re.search(r'(\d+(?:\.\d+)?)mA(\d+)msec-(\d+(?:\.\d+)?)mA(\d+)msec(?:([0-9]+)-([0-9]+)hour)?', file_name)
#     if match:
#         on_current = float(match.group(1))
#         on_time = int(match.group(2))
#         off_current = float(match.group(3))
#         off_time = int(match.group(4))
#         start_hour = int(match.group(5)) if match.group(5) else 0
#         end_hour = int(match.group(6)) if match.group(6) else 0
#         return on_current, on_time, off_current, off_time, start_hour, end_hour
#     else:
#         raise ValueError(f"Filename '{file_name}' does not match expected pattern.")

# # Groups files that are continuations of one another (like 0-24 and 24-48 hour)
# # Use a dictionary to do so, where the keys is the filename without the hours 
# # and the value is a list of tuples in the form of (starthour, 'specific filename')
# # The list of tuples is sorted in chronological order
# def group_files(file_list):
#     groups = defaultdict(list)

#     for file in file_list:
#         try:
#             I1, T1, I2, T2, start, end = extract_parameters(file)
#             key = f"{I1}mA{T1}msec-{I2}mA{T2}msec"
#             groups[key].append((start,file))
#         except ValueError:
#             print("Matching failed for: " + file)
        
#     for key in groups.keys():
#         groups[key].sort(key=lambda x: x[0])

#     return groups

# file_list = [f for f in os.listdir("./rawdata_singlepulse") if f.endswith('.csv')]
# file_groups = group_files(file_list)

# for key, group in file_groups.items():
#     I1, T1, I2, T2, _, _ = extract_parameters(group[0][1])
#     DC = T1/(T1 + T2)
#     T = T1 + T2
#     I_avg = average_current(I1, T1, I2, T2)

#     time_offset = 0
#     time = []
#     voltage = []
#     data_started = False
#     for start, name in group:
#         with open(os.path.join("./rawdata_singlepulse", name),'r') as f:
#             reader = csv.reader(f)
#             for row in reader:
#                 if not any(row):
#                     continue  # skip empty rows

#                 # Extract sample interval
#                 if "Sample interval:" in row[0]:
#                     try:
#                         sample_interval = float(row[0].split(":")[1].strip())
#                     except:
#                         print("Sample interval not retrieved")
#                         pass

#                 # Detect start of data
#                 if row[0].startswith("Enable:"):
#                     data_started = True
#                     continue 

#                 # Parse actual data rows
#                 if data_started:
#                     try:
#                         t = float(row[0]) * sample_interval + time_offset
#                         v = float(row[1])
#                         time.append(t / 3600)
#                         voltage.append(v)
#                     except:
#                         continue

#         _,_,_,_,start,end = extract_parameters(name)
#         time_offset = (end - start) * 3600 + time_offset #offset the timescale for the next sequence of data

#     time_arr = np.array(time)
#     voltage_arr = np.array(voltage)
#     capacity_arr = I_avg * time_arr
#     try:
#         voltage_arr = savgol_filter(voltage_arr, window_length=15501, polyorder=3)
#     except Exception as e:
#         print(f"[SKIPPED] {key}.csv: smoothing failed in stage 1 → {e}")
#         continue

#     output_path = os.path.join("/Users/tomhuang/Documents/battery-power-management-repo/processed_data", f"{key}.csv")
#     with open(output_path, "w", newline = "") as f:
#         writer = csv.writer(f)
#         writer.writerow(["Capacity_mAh", "Voltage_V"])
#         writer.writerows(zip(capacity_arr, voltage_arr))

# #smooth data
# files_list = [f for f in os.listdir("./processed_data") if f.endswith(".csv")]
# fev = 2.2

# for file in files_list:
#     capacity = []
#     voltage = []
#     data_start = False
#     with open(os.path.join("./processed_data",file),'r') as f:
#         reader = csv.reader(f)
#         for row in reader:
#            if row[0].startswith("Capacity_mAh"):
#                data_start = True
#                continue
           
#            if data_start:
#                capacity.append(float(row[0]))
#                voltage.append(float(row[1]))

#     capacity = np.array(capacity)
#     voltage = np.array(voltage)
    
#     #voltage = savgol_filter(voltage, window_length=15501, polyorder=3)

#     mask = (voltage > fev) 
#     capacity = capacity[mask]
#     voltage = voltage[mask] 

#     voltage = voltage / voltage[0]

#     output_path = os.path.join("/Users/tomhuang/Documents/battery-power-management-repo/smoothed_normalized_data", file)
#     with open(output_path, "w", newline = "") as f:
#         writer = csv.writer(f)
#         writer.writerow(["Capacity_mAh", "Voltage_V"])
#         writer.writerows(zip(capacity, voltage))

