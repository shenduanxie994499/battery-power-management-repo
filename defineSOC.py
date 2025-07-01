'''
ULPmodels
    - ULPTechniques
        - VDD Tuning, Power Gating, Clock Gating, DVFS.. etc

MINOTAURclass
    - Hardware Module [SUPERMODULE]
        - CPU, NN Accelerator, RRAM Bank, RRAM Array, SRAM Module, PMU [SUBMODULES]
        - strategies <- ULPTechnique of choice
    - MinotaurChip [CHIP MAIN CLASS]
        - All necessary HardwareModules

# 'strategies' parameter for implementing desired ULP technique for given module
# Base class for all hardware modules
class HardwareModule(ABC):
    def __init__(self,
                 name: str,
                 P_dyn: float,        # dynamic power when active (W)
                 P_stat: float,       # static leakage power (W)
                 Vdd: float,          # supply voltage (V)
                 freq: float,         # operating frequency (Hz)
                 area: float,         # silicon area (mm^2)
                 strategies: list[ULPTechnique] = None):    # list of ULP techniques?
                 # 1 optimization technique per module makes more sense?
        
        self.name = name
        self.P_dyn = P_dyn
        self.P_stat = P_stat
        self.Vdd    = Vdd
        self.freq   = freq
        self.area   = area
        self.strategies = strategies or []

    def base_power(self) -> float:
        """
        Unoptimized power: static + dynamic
        """
        return self.P_stat + self.P_dyn

    def optimized_power(self) -> float:
        """
        Apply each ULPStrategy in sequence to reduce power.
        """
        power = self.base_power()
        for strat in self.strategies:
            power = strat.adjust_power(self, power)
        return power

    @abstractmethod
    def summary(self) -> str:
        """Return a one-line description of the module and its stats."""
        pass

'''

import numpy as np
import matplotlib.pyplot as plt
from HardwareModel import Chip, Module
from ULPmodels import powerGating

chip = Chip("MINOTAUR")

RRAMBANK1 = powerGating(
    DC = 0.3,
    T = 20,
    P_dyn=10.0,        # watts when on
    P_stat_on=1.0,     # leakage when on
    P_stat_off=0.05,   # residual leakage when off
    Vdd_nom=1.0        # volts
)

RRAMBANK2 = powerGating(
    DC = 0.5,
    T = 20,
    P_dyn=10.0,        # watts when on
    P_stat_on=1.0,     # leakage when on
    P_stat_off=0.05,   # residual leakage when off
    Vdd_nom=1.0        # volts
)

rram1 = Module("RRAM", RRAMBANK1)
rram2 = Module("RRAM", RRAMBANK2)
chip.add(rram1)
chip.add(rram2)

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

