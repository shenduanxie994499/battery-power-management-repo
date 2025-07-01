from ULPmodels import ULPTechnique
import numpy as np
import cvxpy as cp

class Module:
    """
    Base class for any modules.
    Defines name, selected ULP technique, and corresponding I, DC, and T (used in the ULPTechniques class).
    Check ULPTechniques class for more details on each of the parameters.
    """
    def __init__(self,
                 name: str,
                 technique: ULPTechnique,):

        self.name = name
        self.technique = technique
    
    def __repr__(self):
        return self.name
    
    def generate_waveform(self, time_array: np.ndarray) -> np.ndarray:
        """
        Build a square‐pulse current draw:
          • On‐time   = DC * T  (current = I)
          • Off‐time  = (1−DC) * T (current = 0)
        time_array in same units as T.

        If I, DC, T are still symbolic cvxpy objects you must have solved a problem
        or set them to plain floats (e.g. via .value or by overriding with numbers)
        before calling this.
        """
        # extract numeric values (fall back if they're already floats)
        try:
            I, DC, T = self.technique.get_IDCT()
        except Exception:
            raise RuntimeError(f"Module {self.name}: need numeric I/DC/T before waveform")

        # square‐pulse: 1 during on‐interval, 0 otherwise
        phase = np.mod(time_array, T)
        on_mask = (phase < DC * T)
        return I * on_mask.astype(float)

class superModule(Module):
    """
    Base class for a superModule, which inherits the Module class.
    Defines name and the modules/supermodules the supermodule contains.
    Included method to return all the modules/supermodules the given supermodule contains.
    """
    def __init__(self,
                 name: str):
        self.name = name
        self.components = []

    def add(self, 
            module: Module):
        self.components.append(module) 
    
    def get_components(self):
        all_components = []
        for component in self.components:
            if type(component) == superModule:
                all_components.append(component.get_components())
            else:
                all_components.append(component)
        return all_components
    
    def combined_waveform(self, time_array: np.ndarray) -> np.ndarray:
        """
        Sum the waveforms of all modules over the same time base.
        """
        if not self.components:
            return np.zeros_like(time_array)
        stacked = np.zeros_like(time_array, dtype=float)
        for m in self.components:
            stacked += m.generate_waveform(time_array)
        return stacked

class Chip(superModule):
    """
    Base class for a chip, which inherits the superModule class.
    Defines name, will use methods inherited from superModule class.
    """
    def __init__(self,
                 name: str):
        self.name = name
        self.components = []