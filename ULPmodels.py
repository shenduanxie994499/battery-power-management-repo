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
    def __init__(self,
                DC: float, 
                T: float,
                P_dyn: float,
                P_stat_on: float,
                P_stat_off: float,
                V_dd: float):
        self.P_dyn = P_dyn
        self.P_stat_on = P_stat_on
        self.P_stat_off = P_stat_off
        self.V_dd = V_dd
        self.DC = DC
        self.T  = T
        # I is calculated given device power metrics
        self.I = self.get_expressionI()
        self.DCvar = cp.Variable(name="DC")
        self.Tvar  = cp.Variable(name="T", nonneg=True)

        self.bounds = [
            self.DC >= 0, self.DC <= 1,
            self.T >= 0
        ]

    def get_expressionI(self):
        return (self.P_stat_on * (self.DC) + self.P_dyn * (self.DC) + self.P_stat_off * (1 - self.DC)) / self.V_dd
    
    def get_IDCT(self):
        return self.I, self.DC, self.T

    # @abstractmethod
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
                 DC: float,
                 T: float,
                 P_dyn: float,
                 P_stat: float,
                 V_dd: float,
                 perf_ips: float,
                 f_clock:  float):
        super().__init__(DC, T, P_dyn, P_stat, P_stat, V_dd)

        # Convert your known power metrics into a fixed peak‐current:
        # self.I_idle = P_stat / Vdd      # not used?
        # self.I = (P_dyn + P_stat + P_sc) / Vdd
        self.I = self.get_expressionI()
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
                DC: float, 
                T: float,
                P_dyn: float,
                P_stat_on: float,
                P_stat_off: float,
                Vdd_nom: float   # supply voltage (V)
                ):
        super().__init__(DC, T, P_dyn, P_stat_on, P_stat_off, Vdd_nom)
        self.I = self.get_expressionI()
        # NOTE: There technically is some residual leakage while gated area is off. How is this calculated?
    
class clockGating(ULPTechnique):
    """
    Crude implementation of clock gating of a circuit block.
    Takes into account subthrehsold leakage current and static power assumed to be given (unlike power gating, V_dd and GND is still connected in clock gated blocks).
    Missing implementation of power and delay penalty during wake-up of the block.
    """
    def __init__(self,
                 DC: float,
                 T: float,
                 P_dyn: float,  # dynamic power of full block (W)
                 P_stat: float,  # static (leakage) power of full block (W)
                 V_dd: float   # supply voltage (V)
                 ):
        super().__init__(DC, T, P_dyn, P_stat, P_stat, V_dd)
        self.I = self.get_expressionI()

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
                 DC: float,
                 T: float,
                 P_dyn: float,
                 P_stat: float,
                 V_dd: float,
                 f_clock: float,
                 perf_ips: float):
        super().__init__(DC, T, P_dyn, P_stat, P_stat, V_dd)
        self.I = (P_dyn + P_stat) / V_dd
        self._perf_cons = [
            self.DC * f_clock >= perf_ips
        ]
    
