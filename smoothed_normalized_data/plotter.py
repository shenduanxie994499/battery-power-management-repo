import os
import csv
import matplotlib.pyplot as plt
import re
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# === CONFIG ===
data_dir = "/Users/tomhuang/Documents/battery-power-management-repo/smoothed_normalized_data"
csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

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
    
def average_current(on_current, on_time, off_current, off_time):
    return (on_current * on_time + off_current * off_time) / (on_time + off_time)

duty_cycles = []
avg_currents = []
for f in csv_files:
    I1, T1, I2, T2, _, _ = extract_parameters(f)
    DC = T1 / (T1 + T2)
    I_avg = average_current(I1, T1, I2, T2)
    if DC is not None:
        duty_cycles.append(DC)
    if I_avg is not None:
        avg_currents.append(I_avg)

norm = mcolors.Normalize(vmin=min(avg_currents), vmax=max(avg_currents))
cmap = cm.viridis
    
duty_min = min(duty_cycles)
duty_max = max(duty_cycles)

def get_dash_pattern(dc):
    # Normalize
    norm_dc = (dc - duty_min) / (duty_max - duty_min + 1e-6)
    dash_len = 5 + 10 * norm_dc     # longer dashes for higher duty
    gap_len = 15 - 10 * norm_dc     # shorter gaps for higher duty
    return (0, (dash_len, max(gap_len, 1)))

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
        I1, T1, I2, T2, _, _ = extract_parameters(file_name)
        DC = T1 / (T1 + T2)
        Iavg = average_current(I1, T1, I2, T2)
        color = cmap(norm(Iavg))

        if DC is None:
            continue
        dash = get_dash_pattern(DC)
        label = f"DC={DC:.2f}"
        ax.plot(capacity, voltage, label=file_name.replace(".csv", ""),linestyle = dash,color=color)

# === LABELS ===
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # required in some matplotlib versions
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label("Average Current (mA)")

ax.set_xlabel("Capacity (mAh)")
ax.set_ylabel("Normalized Voltage (a.u.)")
ax.set_title("Raw Discharge Curves from CSV Files")
ax.legend(fontsize="x-small", loc="best")
ax.grid(True)
# ax.set_xlim(0,50)
# ax.set_ylim(2,3.5)
plt.tight_layout()

# === SHOW ===
plt.show()
