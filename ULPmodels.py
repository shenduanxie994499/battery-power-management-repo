import cvxpy as cp
from abc import ABC, abstractmethod

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
