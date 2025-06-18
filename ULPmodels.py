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
        # self.I  = cp.Variable(name="I", nonneg=True)
        # I is calculated given device power metrics
        self.DC = cp.Variable(name="DC")
        self.T  = cp.Variable(name="T", nonneg=True)

        self.bounds = [
            self.DC >= 0, self.DC <= 1,
            self.T >= 0
        ]

    def decision_vars(self):
        return [self.I, self.DC, self.T]

    @abstractmethod
    def capacity_expr(self):
        """
        Return a CVXPY expression for effective battery capacity
        as a function of (I, DC, T).
        """
        # Convex optimization library CVXPY gives mathematical object for optimization algo
        # idk how we plan to optimize
        pass

    def all_constraints(self):
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

    def get_IDCT(self):
        # feed the **fixed** I_peak + your DC & T into Shen’s pulse model:
        return self.I, self.DC, self.T
        
    def constraints(self):
        return super().constraints()

"""
Notes/Questions:
    - We need constraints for ensuring performance doesn't drop right?
    - Dynamic and static power is given. To clarify, we aren't actually trying to optimize for power right?
    We are optimizing for battery life?
    - Will P_dynamic and P_static be given or I_dynamic and I_static?
"""

class powerGating(ULPTechnique):
    """
    Crude implementation of power gating of a circuit block. 
    Uses the subthreshold current equation from "Energy optimality and variability in subthreshold design" (Hanson et al), 
    which requires a lot of information of the transistor/circuit block.
    Missing implementation of power and delay penalty during wake-up of the block.
    """
    def __init__(self,
                 V_gs: float,   # Gate to source voltage
                 W: float,      # Width of device
                 L: float,      # Length of device
                 mu_eff: float, # Effective mobility
                 C_ox: float,   # Oxide capacitance of grid
                 m: float,      # Slope coefficient of the current below sub-threshold,
                 v_T: float,    # Thermal voltage kT/q
                 V_th: float,   # Threshold voltage
                 V_ds: float    # Drain to source voltage
                 ):
        super().__init__()
        self.I = (W/L) * mu_eff * C_ox * (m-1) * np.power(v_T, 2) * np.exp((V_gs - V_th) / (m * v_T)) * (1 - np.exp(-V_ds / v_T)) # Equivalent to subthreshold leakage current

    def constraints(self):
        return super().constraints()

    def get_IDCT(self):
        # feed the **fixed** I_peak + your DC & T into Shen’s pulse model:
        return self.I, self.DC, self.T
    
class clockGating(ULPTechnique):
    """
    Crude implementation of clock gating of a circuit block.
    Takes into account subthrehsold leakage current and static power assumed to be given (unlike power gating, V_dd and GND is still connected in clock gated blocks).
    Missing implementation of power and delay penalty during wake-up of the block.
    """
    def __init__(self,
                 V_gs: float,   # Gate to source voltage
                 W: float,      # Width of device
                 L: float,      # Length of device
                 mu_eff: float, # Effective mobility
                 C_ox: float,   # Oxide capacitance of grid
                 m: float,      # Slope coefficient of the current below sub-threshold,
                 v_T: float,    # Thermal voltage kT/q
                 V_th: float,   # Threshold voltage
                 V_ds: float,   # Drain to source voltage
                 P_stat: float, # Static Power
                 V_dd: float    # Power supply voltage
                 ):
        super().__init__()
        self.I_sub = (W/L) * mu_eff * C_ox * (m-1) * np.power(v_T, 2) * np.exp((V_gs - V_th) / (m * v_T)) * (1 - np.exp(-V_ds / v_T))
        self.I_idle = P_stat / V_dd
        self.I = self.I_sub + self.I_idle
    
    def constraints(self):
        return super().constraints()

    def get_IDCT(self):
        # feed the **fixed** I_peak + your DC & T into Shen’s pulse model:
        return self.I, self.DC, self.T

