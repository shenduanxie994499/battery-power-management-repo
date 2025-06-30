import cvxpy as cp
from abc import ABC, abstractmethod
import numpy as np
import random
import matplotlib.pyplot as plt

class ULPTechnique(ABC):
    """
    Base ABSTRACT class for any ULP technique.
    Defines three variables proposed in Duanxie's paper:
      - I  : peak discharge current (A)
      - DC : duty cycle
      - T  : pulse period (s)
    """
    def __init__(self,
                  I: float,
                  DC: float, 
                  T: float):
        self.I  = I
        # I is calculated given device power metrics
        self.DC = DC
        self.T  = T

        self.bounds = [
            self.DC >= 0, self.DC <= 1,
            self.T >= 0,
            self.I >= 0
        ]

    def get_IDCT(self):
        return self.I, self.DC, self.T

    def constraints(self):
        """
        Gather any additional constraints (e.g. Vdd bounds, performance, etc.).
        Subclasses can extend by appending to self.bounds.
        """
        return list(self.bounds)

# Reduce VDD to decrease power losses
class VDDTuning(ULPTechnique):
    """
    If you know your device’s:
      • P_dyn_active  — dynamic power when “on” (W)
      • P_stat_active — static leakage power when “on” (W)
      • P_stat_idle   — static leakage power when “off” (W)
    then you can precompute your peak current:
      I_peak = (P_dyn_active + P_stat_active) / Vdd_nom
    (Vdd_nom is just the known supply voltage; not a decision var.)
    """
    def __init__(self,
                 P_dyn: float,
                 P_stat: float,
                 P_sc: float,       # In practice, I believe this is lumped in with dynamic power?
                 Vdd: float,
                 perf_ips: float,
                 f_clock:  float):
        # super().__init__()

        # Convert your known power metrics into a fixed peak‐current:
        self.I_idle = P_stat / Vdd      # not used?
        self.I = (P_dyn + P_stat + P_sc) / Vdd

        # Performance constraint: while “on” (DC fraction), you need enough throughput
        #   DC * f_clock ≥ perf_ips
        self._perf_cons = [
            self.DC * f_clock >= perf_ips
        ]

    def constraints(self):
        return super().constraints() + self._perf_cons

class powerGating(ULPTechnique):
    """
    Crude implementation of power gating of a circuit block. 
    Uses the subthreshold current equation from "Energy optimality and variability in subthreshold design" (Hanson et al), 
    which requires a lot of information of the transistor/circuit block.
    Missing implementation of power and delay penalty during wake-up of the block.
    """
    def __init__(self,
                 P_dyn: float,  # dynamic power of full block (W)
                 P_stat: float,  # static (leakage) power of full block (W)
                 frac_gated: float,  # fraction of block actually gated off
                 Vdd_nom: float   # supply voltage (V)
                 ):
        # super().__init__()
        #self.I = (W/L) * mu_eff * C_ox * (m-1) * np.power(v_T, 2) * np.exp((V_gs - V_th) / (m * v_T)) * (1 - np.exp(-V_ds / v_T)) # Equivalent to subthreshold leakage current
        self.I = (P_stat * (1 - frac_gated) + P_dyn * (1 - frac_gated)) / Vdd_nom
        # NOTE: There technically is some residual leakage while gated area is off. How is this calculated?
    
class clockGating(ULPTechnique):
    """
    Crude implementation of clock gating of a circuit block.
    Takes into account subthrehsold leakage current and static power assumed to be given (unlike power gating, V_dd and GND is still connected in clock gated blocks).
    Missing implementation of power and delay penalty during wake-up of the block.
    Note: static power for clock gating is typically higher than power gating, so this could be a constraint that can be added.
    """
    def __init__(self,
                 P_dyn: float,  # dynamic power of full block (W)
                 P_stat: float,  # static (leakage) power of full block (W)
                 frac_gated: float,  # fraction of block that is clock-gated
                 Vdd_nom: float   # supply voltage (V)
                 ):
        # super().__init__()
        self.I = (P_stat * (1 - frac_gated) + P_dyn * (1 - frac_gated)) / Vdd_nom

class DVFS(ULPTechnique):
    """
    Dynamic Voltage and Frequency Scaling. Assume 5 phases:
    Phase 1: Detection via analog sensors and other peripherals
    Phase 2: Processing data with processor
    Phase 3: Computation of data with processor/NN Accelerator
    Phase 4: Transmission of data
    Phase 5: Idle
    Phase 1 and 4 can have lower frequency/voltage. Phase 2 and 3 should have higher frequency and voltage to finish the task. Phase 5 should have lowest freq./voltage.
    For now, the user sets the voltage and frequency limits. In the future, perhaps the class can integrate the freq./volt. limits internally.

    Typically, there are a number of set frequencies that DVFS can toggle between to optimize for power and performance.
    This model should probably have the option for a list of frequencies to choose from and calculate power loss for each frequency state.
    """
    def __init__(self,
                 phase: int,
                 alpha: float,
                 C_tot: float,
                 I_leak: float,
                 V_dd: float,
                 f_clock: float,
                 perf_ips: float):
        # super().__init__(self.I, self.DC, self.T)
        self.alpha = alpha
        self.V_dd = V_dd
        self.I_leak = I_leak
        self.C_tot = C_tot
        self.f_clock = f_clock 

        self.P_stat = I_leak * self.V_dd
        self.P_dyn = alpha * C_tot * (self.V_dd ** 2) * f_clock
        self.I = (self.P_dyn + self.P_stat) / V_dd
        """self._perf_cons = [
            self.DC * f_clock >= perf_ips
        ]"""
        if self.P_stat > self.P_dyn:
            raise ValueError("P_stat greater than P_dyn")
    
    def f(self):
        # Does some calculation to return a runtime?
        # Right now it will be a random value from 0 to 1
        return random.random()

    def voltage_scale(self):
        """
        Assumes lower bound is when static power becomes less than dynamic power.
        Assumes upper bound is when static power becomes less than 10% of dynamic power.
        Return lower/upper bound of V_dd as a list.
        """
        P_stat = 0
        P_dyn = 0
        V_dd = 0
        voltageLimit = []
        while True:
            V_dd += 0.01
            P_stat = self.I_leak * V_dd
            P_dyn = self.alpha * self.C_tot * (V_dd ** 2) * self.f_clock
            if P_stat <= P_dyn:
                if len(voltageLimit) == 0:
                    voltageLimit.append(V_dd)
                    continue
                if P_stat * 9 < P_dyn:
                    voltageLimit.append(V_dd)
                    break
        
        return np.round(voltageLimit, 2)
    
    def frequency_scale(self):
        """
        Assumes lower bound is when static power becomes less than dynamic power.
        Assumes upper bound is when static power becomes less than 10% of dynamic power.
        Return lower/upper bound of f_clock as a list.
        """
        P_stat = self.I_leak * self.V_dd
        P_dyn = 0
        f_clock = 0
        freqLimit = []
        while True:
            f_clock += 10**6
            P_dyn = self.alpha * self.C_tot * (self.V_dd ** 2) * f_clock
            if P_stat <= P_dyn:
                if len(freqLimit) == 0:
                    freqLimit.append(f_clock)
                    continue
                if P_stat * 9 < P_dyn:
                    freqLimit.append(f_clock)
                    break
        return np.round(freqLimit, 2)

"""DVS = DVFS(1, 0.9, 20*(10**-9), 20*(10**-3), 1.2, 100*(10**6), 0)
DVS.I = 0
DVS.DC=0
DVS.T = 0
print(DVS.frequency_scale())"""
