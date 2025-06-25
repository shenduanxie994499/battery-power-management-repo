import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import re

#README: We understand that the single pulse discharge charactersitics
# is under the joint effect of duty cycle and pulse period. In contrast to previous
# analysis, we fix the time for discharge and change the time for idle. 
#Refer to section 2.2.4 in the thesis

# File I/O definitions: 10,15,25,50,75 stands for waveform period
csv_files_10 = ['rawdata_singlepulse/' + f for f in [
    '30mA1msec9msec0-24hour.csv',
    '30mA1msec9msec24-48hour.csv',
    '30mA1msec9msec48-72hour.csv'
]]

csv_files_15 = ['rawdata_singlepulse/' + f for f in [
    '30mA1msec14msec0-24hour.csv',
    '30mA1msec14msec24-48hour.csv',
    '30mA1msec14msec48-72hour.csv',
    '30mA1msec14msec72-84hour.csv'
]]

csv_files_25 = ['rawdata_singlepulse/' + f for f in [
    '30mA1msec24msec0-24hour.csv',
    '30mA1msec24msec24-48hour.csv',
    '30mA1msec24msec48-72hour.csv',
    '30mA1msec24msec72-84hour.csv',
    '30mA1msec24msec84-252hour.csv'
]]

csv_files_50 = ['rawdata_singlepulse/' + f for f in [
    '30mA1msec49msec0-24hour.csv',
    '30mA1msec49msec24-48hour.csv',
    '30mA1msec49msec48-72hour.csv',
    '30mA1msec49msec84-252hour.csv'
]]

csv_files_75 = ['rawdata_singlepulse/' + f for f in [
    '30mA1msec74msec0-24hour.csv',
    '30mA1msec74msec24-48hour.csv',
    '30mA1msec74msec48-72hour.csv',
    '30mA1msec74msec84-252hour.csv'
]]

# Constants
short_sample_time = 0.5  # seconds/sample: the time duration between two samples;
long_sample_time = 5.0   # seconds/sample: for several rawdara files, the time duration between two samples is 5 seconds (check csv files)
off_current = 0.2        # mA (idle current)
on_current = 30.0        # mA (active pulse current)

# Function to extract parameters from file name
def extract_parameters(file_name):
    """
    Extract on_time, off_time, start_hour, end_hour from file name.
    Example: '30mA1msec24msec0-24hour.csv' → 1, 24, 0, 24
    """
    match = re.search(r'(\d+)mA(\d+)msec(\d+)msec(\d+)-(\d+)hour', file_name)
    if match:
        on_time = int(match.group(2))
        off_time = int(match.group(3))
        start_hour = int(match.group(4))
        end_hour = int(match.group(5))
        return on_time, off_time, start_hour, end_hour
    else:
        raise ValueError(f"Filename '{file_name}' does not match expected pattern.")

# function to calculate average discharge current based on discharge waveform
def average_current_function(on_current, on_time, off_current, off_time):
    return (on_current * on_time + off_current * off_time) / (on_time + off_time)

# Function to read CSV files and extract discahrge charactersitics (curves)
def csv_read_function(csv_files):
    time_data = []
    capacity_data = []
    voltage_data = []
    
    total_time_offset = 0  # Accumulates time in seconds

    for file in csv_files:
        on_time, off_time, start_hour, end_hour = extract_parameters(file)
        sample_time = long_sample_time if '84-252hour' in file else short_sample_time
        avg_current = average_current_function(on_current, on_time, off_current, off_time)

        with open(file, 'r') as f:
            reader = csv.reader(f)
            for row_index, row in enumerate(reader):
                if len(row) < 2:
                    continue
                try:
                    temp_t = float(row[0]) * sample_time + total_time_offset
                    temp_v = float(row[1])
                    time_data.append(temp_t)
                    voltage_data.append(temp_v)
                except ValueError:
                    continue

        # Update offset using actual duration (in seconds)
        file_duration_sec = (end_hour - start_hour) * 3600
        total_time_offset += file_duration_sec

    time_data = [x / 3600 for x in time_data]  # seconds → hours
    #data filtering
    filtered_time_data = [t for t,v in zip(time_data,voltage_data) if (t<168 and (2.6<v<3.0))]
    filtered_voltage_data = [v for t,v in zip(time_data,voltage_data) if (t<168 and (2.6<v<3.0))]
    filtered_capacity_data = [x * avg_current for x in filtered_time_data]
    return filtered_time_data, filtered_capacity_data, filtered_voltage_data

# read csv functions for characteristics (curves) of different pulse periods
time_10, capacity_10, voltage_10 = csv_read_function(csv_files_10)
time_15, capacity_15, voltage_15 = csv_read_function(csv_files_15)
time_25, capacity_25, voltage_25 = csv_read_function(csv_files_25)
time_50, capacity_50, voltage_50 = csv_read_function(csv_files_50)
time_75, capacity_75, voltage_75 = csv_read_function(csv_files_75)


#regression analysis to be implemented (refer to other scripts)


# Plotting the results
plt.figure()
plt.plot(capacity_10, voltage_10, 'b.', label='10 msec IDLE')
plt.plot(capacity_15, voltage_15, 'g.', label='15 msec IDLE')
plt.plot(capacity_25, voltage_25, 'r.', label='25 msec IDLE')
plt.plot(capacity_50, voltage_50, 'y.', label='50 msec IDLE')
plt.plot(capacity_75, voltage_75, 'c.', label='75 msec IDLE')
plt.xlabel('Capacity (mAh)')
plt.ylabel('Voltage (V)')
plt.title('Binary-pulsed discharge curve in duty cycle and pulse period joint tests')
plt.legend()
plt.grid(True)
plt.show()
