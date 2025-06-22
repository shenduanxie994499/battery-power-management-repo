import csv
import os
import matplotlib.pyplot as plt

# List of CSV files to read (update with your actual file names or a folder scan)
csv_files = ['5mA1msec-0.2mA24msec0-12hours.csv', '5mA1msec-0.2mA24msec12-24hours.csv', '5mA1msec-0.2mA24msec24-48hours.csv',
             '5mA1msec-0.2mA24msec48-72hours.csv','5mA1msec-0.2mA24msec72-96hours.csv']

# Initialize combined arrays
column1 = []
column2 = []

last_value = 0  # Starting point for continuity

for i, filename in enumerate(csv_files):
    if os.path.isfile(filename):
        with open(filename, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header if present

            local_col1 = []
            local_col2 = []

            for row in reader:
                if len(row) >= 2:
                    try:
                        x = float(row[0])
                        y = float(row[1])
                        local_col1.append(x)
                        local_col2.append(y)
                    except ValueError:
                        # Skip rows that can't be converted to float
                        continue

            # Offset the local column1 values by the last value
            if local_col1:
                if i > 0:
                    # Shift all values by last_value
                    local_col1 = [x + last_value for x in local_col1]

                last_value = local_col1[-1]  # Update last value for next file

                # Add to the global arrays
                column1.extend(local_col1)
                column2.extend(local_col2)

                 # Sample one pair for every 100 entries
                sampled_col1 = local_col1[::1000]
                sampled_col2 = local_col2[::1000]

                column1.extend(sampled_col1)
                column2.extend(sampled_col2)
#adjust to voltage reponses in time domain. each sampling point stands for 0.5sec
time = [0.5*x/3600 for x in column1]
voltage = column2

#filter: remove noise points, custom design here
constant1 = 38  #time before 172800
constant2 = 2.895 #voltage smaller than 2.8
filtered_time = [x for x, y in zip(time, voltage) if not (x < constant1 and y < constant2)]
filtered_voltage = [y for x, y in zip(time, voltage) if not (x < constant1 and y < constant2)]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(filtered_time, filtered_voltage, marker='o', linestyle='', color='blue')
plt.xlabel("time (hour)")
plt.ylabel("voltage (volt)")
plt.title("Inital voltage drop in discharge curve")
plt.grid(True)
plt.tight_layout()
plt.show()

