import yaml
import HardwareModel as hm
import ULPmodels as model
import numpy as np
import matplotlib.pyplot as plt

with open('test.yaml', 'r') as f:
    chip_data = yaml.safe_load(f)

modules = {}
supermodules = {}
chips = {}

for section in chip_data:
    if section == "Module":
        for item in chip_data[section]:
            if item["technique"] == "VDDTuning":
                modules[item["name"]] = hm.Module(item["name"], model.VDDTuning(item["DC"], item["T"], item["P_dyn"], item["P_stat_on"], item["V_dd"], item["f_clock"], item["perf_ips"]))
            
            if item["technique"] == "clockGating":
                modules[item["name"]] = hm.Module(item["name"], model.clockGating(item["DC"], item["T"], item["P_dyn"], item["P_stat_on"], item["V_dd"]))
            
            if item["technique"] == "powerGating":
                modules[item["name"]] = hm.Module(item["name"], model.powerGating(item["DC"], item["T"], item["P_dyn"], item["P_stat_on"], item["P_stat_off"], item["V_dd"]))
            
            if item["technique"] == "DVFS":
                modules[item["name"]] = hm.Module(item["name"], model.DVFS(item["phase"], item["DC"], item["T"], item["P_dyn"], item["P_stat_on"], item["V_dd"], item["f_clock"], item["perf_ips"]))
    
    if section == "superModule":
        for item in chip_data[section]:
            supermodules[item["name"]] = hm.superModule(item["name"])
            if item['modules'] is not None:
                for component in item["modules"]:
                    supermodules[item["name"]].add(modules[component])

            if item['supermodules'] is not None:
                for component in item["supermodules"]:
                    supermodules[item["name"]].add(supermodules[component])
    
    if section == "Chip":
        for item in chip_data[section]:
            chips[item["name"]] = hm.Chip(item["name"])
            if item['modules'] is not None:
                for component in item["modules"]:
                    chips[item["name"]].add(modules[component])
            
            if item['supermodules'] is not None:
                for component in item["supermodules"]:
                    chips[item["name"]].add(supermodules[component])


chip = chips["test"]
t = np.linspace(0, 100, 1001)   # 0…100 s, 1001 points
current = chip.combined_waveform(t)

plt.figure(figsize=(8,3))
plt.plot(t, current, lw=2)
plt.xlabel("Time (s)")
plt.ylabel("Total Current (A)")
plt.title("Stacked Pulse Waveform for RRAM Module")
plt.grid(True)
plt.tight_layout()
plt.show()