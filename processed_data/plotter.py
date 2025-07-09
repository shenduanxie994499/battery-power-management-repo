import os
import csv
import matplotlib.pyplot as plt

# === CONFIG ===
data_dir = "/Users/tomhuang/Documents/battery-power-management-repo/processed_data"
csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

plt.close('all')              
fig, ax = plt.subplots()      

for file_name in csv_files:
    file_path = os.path.join(data_dir, file_name)
    capacity = []
    voltage = []

    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 2:
                try:
                    c = float(row[0])
                    v = float(row[1])
                    capacity.append(c)
                    voltage.append(v)
                except:
                    continue

    if capacity and voltage:
        ax.plot(capacity, voltage, label=file_name.replace(".csv", ""))

# === LABELS ===
ax.set_xlabel("Capacity (mAh)")
ax.set_ylabel("Voltage (V)")
ax.set_title("Raw Discharge Curves from CSV Files")
ax.legend(fontsize="x-small", loc="best")
ax.grid(True)
# ax.set_xlim(0,50)
# ax.set_ylim(2,3.5)
plt.tight_layout()

# === SHOW ===
plt.show()
