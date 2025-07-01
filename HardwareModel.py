from ULPmodels import ULPTechnique, powerGating
import cvxpy as cp
import random
import matplotlib.pyplot as plt

class Module:
    """
    Base class for any modules.
    Defines name, selected ULP technique, and corresponding I, DC, and T (used in the ULPTechniques class).
    Check ULPTechniques class for more details on each of the parameters.
    """
    def __init__(self,
                 name: str,
                 technique: ULPTechnique):

        self.name = name
        self.technique = technique
    
    def power_consumption(self):
        return self.technique.P_dyn + self.technique.P_stat

    def __repr__(self):
        return self.name
    
    def generate_waveform(self, t_total):
        t_on = self.technique.DC * self.technique.T
        t_off = self.technique.T - t_on
        cycles = int(t_total // self.technique.T)
        print(t_on, t_off)
        time = [0.0]
        I_stat = self.technique.P_stat / self.technique.V_dd
        I_dyn = self.technique.I
        current = [I_stat]

        t = 0
        for i in range(cycles):
            t += t_off
            time.append(t)
            current.append(I_dyn)
            t += t_on
            time.append(t)
            current.append(I_stat)

        plt.step(time, current, where='post')
        plt.show()

class SuperModule(Module):
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
            if type(component) == SuperModule:
                all_components.append(component.get_components())
            else:
                all_components.append(component)
        return all_components

    def __repr__(self):
        return f"Supermodule: {self.name}\nComponents: {self.get_components()[0]}"

class Chip(SuperModule):
    """
    Base class for a chip, which inherits the superModule class.
    Defines name, will use methods inherited from superModule class.
    """
    def __init__(self,
                 name: str):
        self.name = name
        self.components = []
    
    def __repr__(self):
        return f"Chip: {self.name}\nComponents: {self.get_components()[0]}"


T = 1/(10**8) # 100 MHz
f_clk = 10**8 # 100 MHz
DC = random.random()
task_time = DC * T
rram1_powergate = powerGating(10**(-2), 10**(-3), 0.5, 0.8, DC, T)
rram1 = ULPTechnique(DC, T, 10**(-2), 10**(-3), 0.8)
RRAM_Bank_1 = Module("RRAM_Bank_1", rram1)
rram1_gated = Module('rr', rram1_powergate)
RRAM_Array_1 = SuperModule("RRAM_Bank_1")
RRAM_Array_1.add(RRAM_Bank_1)
MINOTAUR = Chip("MINOTAUR")
MINOTAUR.add(RRAM_Array_1)
RRAM_Bank_1.generate_waveform(2*10**(-8))
rram1_gated.generate_waveform(2*10**(-8))