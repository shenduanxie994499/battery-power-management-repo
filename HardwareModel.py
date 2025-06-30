from ULPmodels import ULPTechnique, powerGating
import cvxpy as cp

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
    
    def __repr__(self):
        return self.name

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
