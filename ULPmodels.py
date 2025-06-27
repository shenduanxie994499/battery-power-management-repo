import cvxpy as cp
from abc import ABC, abstractmethod
import numpy as np

class ULPTechnique(ABC):
    """
    Base ABSTRACT class for any ULP technique.
    Defines three variables proposed in Duanxie's paper:
      - I  : peak discharge current (A)
      - DC : duty cycle
      - T  : pulse period (s)
    """
    def __init__(self):
        self.I  = cp.Variable(name="I", nonneg=True)
        # I is calculated given device power metrics
        self.DC = cp.Variable(name="DC")
        self.T  = cp.Variable(name="T", nonneg=True)

        self.bounds = [
            self.DC >= 0, self.DC <= 1,
            self.T >= 0
        ]

    def get_IDCT(self):
        return self.I, self.DC, self.T

    @abstractmethod
    def capacity_expr(self):
        """
        Return a CVXPY expression for effective battery capacity
        as a function of (I, DC, T).
        """
        # Convex optimization library CVXPY gives mathematical object for optimization algo
        # idk how we plan to optimize
        pass

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
        super().__init__()

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
                 # V_gs: float,   # Gate to source voltage
                 # W: float,      # Width of device
                 # L: float,      # Length of device
                 # mu_eff: float, # Effective mobility
                 # C_ox: float,   # Oxide capacitance of grid
                 # m: float,      # Slope coefficient of the current below sub-threshold,
                 # v_T: float,    # Thermal voltage kT/q
                 # V_th: float,   # Threshold voltage
                 # V_ds: float    # Drain to source voltage
                 ):
        super().__init__()
        #self.I = (W/L) * mu_eff * C_ox * (m-1) * np.power(v_T, 2) * np.exp((V_gs - V_th) / (m * v_T)) * (1 - np.exp(-V_ds / v_T)) # Equivalent to subthreshold leakage current
        self.I = (P_stat * (1 - frac_gated) + P_dyn * (1 - frac_gated)) / Vdd_nom
        # NOTE: There technically is some residual leakage while gated area is off. How is this calculated?
    
class clockGating(ULPTechnique):
    """
    Crude implementation of clock gating of a circuit block.
    Takes into account subthrehsold leakage current and static power assumed to be given (unlike power gating, V_dd and GND is still connected in clock gated blocks).
    Missing implementation of power and delay penalty during wake-up of the block.
    """
    def __init__(self,
                 P_dyn: float,  # dynamic power of full block (W)
                 P_stat: float,  # static (leakage) power of full block (W)
                 frac_gated: float,  # fraction of block that is clock-gated
                 Vdd_nom: float   # supply voltage (V)
                 # V_gs: float,   # Gate to source voltage
                 # W: float,      # Width of device
                 # L: float,      # Length of device
                 # mu_eff: float, # Effective mobility
                 # C_ox: float,   # Oxide capacitance of grid
                 # m: float,      # Slope coefficient of the current below sub-threshold,
                 # v_T: float,    # Thermal voltage kT/q
                 # V_th: float,   # Threshold voltage
                 # V_ds: float,   # Drain to source voltage
                 # P_stat: float, # Static Power
                 # V_dd: float    # Power supply voltage
                 ):
        super().__init__()
        #self.I_sub = (W/L) * mu_eff * C_ox * (m-1) * np.power(v_T, 2) * np.exp((V_gs - V_th) / (m * v_T)) * (1 - np.exp(-V_ds / v_T))
        #self.I_idle = P_stat / V_dd
        #self.I = self.I_sub + self.I_idle
        I_leak = P_stat / Vdd_nom
        I_dyn  = (P_dyn * (1 - frac_gated)) / Vdd_nom
        self.I = I_leak + I_dyn

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
                 P_dyn: float,
                 P_stat: float,
                 P_sc: float,
                 V_dd: float,
                 f_clock: float,
                 perf_ips: float):
        super().__init__()
        self.I = (P_dyn + P_stat + P_sc) / V_dd
        self._perf_cons = [
            self.DC * f_clock >= perf_ips
        ]
    
