from abc import ABC, abstractmethod
from ULPmodels import ULPTechnique

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
'''
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


class CPU64(HardwareModule):
    """
    64-bit in-order RISC CPU core
    """
    def __init__(self,
                 P_dyn: float,
                 P_stat: float,
                 Vdd: float,
                 freq: float,
                 area: float,
                 ipc: float,
                 strategies: list[ULPTechnique] = None):
        super().__init__("CPU64", P_dyn, P_stat, Vdd, freq, area, strategies)
        self.ipc = ipc  # instructions per cycle

    def summary(self) -> str:
        return f"CPU64 @ {self.freq/1e6:.1f}MHz, IPC={self.ipc}, area={self.area}mm^2"


class NNAccelerator(HardwareModule):
    """
    Neural-network accelerator block
    """
    def __init__(self,
                 P_dyn: float,
                 P_stat: float,
                 Vdd: float,
                 freq: float,
                 area: float,
                 ops_per_cycle: float,
                 strategies: list[ULPTechnique] = None):
        super().__init__("NNAccel", P_dyn, P_stat, Vdd, freq, area, strategies)
        self.ops_per_cycle = ops_per_cycle

    def summary(self) -> str:
        return f"NNAccel @ {self.freq/1e6:.1f}MHz, OPC={self.ops_per_cycle}, area={self.area}mm^2"


class RRAMBank(HardwareModule):
    """
    Single RRAM bank (128b wide, power-gatable)
    """
    def __init__(self,
                 bank_id: int,
                 P_dyn: float,
                 P_stat: float,
                 Vdd: float,
                 freq: float,
                 area: float,
                 is_powered: bool = True,
                 strategies: list[ULPTechnique] = None):
        
        name = f"RRAMBank{bank_id}"
        super().__init__(name, P_dyn, P_stat, Vdd, freq, area, strategies)
        self.bank_id = bank_id
        self.is_powered = is_powered

    def base_power(self) -> float:
        # If gated off, no dynamic or leakage
        if not self.is_powered:
            return 0.0
        return super().base_power()

    def summary(self) -> str:
        state = "ON" if self.is_powered else "OFF"
        return f"RRAMBank{self.bank_id}({state}), area={self.area}mm^2"


class RRAMArray(HardwareModule):
    """
    Full RRAM array composed of multiple banks
    MINOTAUR has 12 banks of 128b RRAM
    """
    def __init__(self,
                 num_banks: int,
                 P_dyn_per_bank: float,
                 P_stat_per_bank: float,
                 Vdd: float,
                 freq: float,
                 area_per_bank: float,
                 strategies: list[ULPTechnique] = None):
        
        super().__init__("RRAMArray",
                         P_dyn_per_bank * num_banks,
                         P_stat_per_bank * num_banks,
                         Vdd,
                         freq,
                         area_per_bank * num_banks,
                         strategies)
        self.banks = [RRAMBank(i,
                               P_dyn_per_bank,
                               P_stat_per_bank,
                               Vdd,
                               freq,
                               area_per_bank,
                               True,
                               strategies)
                      for i in range(num_banks)]

    def base_power(self) -> float:
        return sum(bank.base_power() for bank in self.banks)

    def optimized_power(self) -> float:
        return sum(bank.optimized_power() for bank in self.banks)

    def summary(self) -> str:
        return f"RRAMArray: {len(self.banks)} banks, total area={self.area}mm^2"


class SRAMModule(HardwareModule):
    """
    SRAM memory block
    """
    def __init__(self,
                 P_dyn: float,
                 P_stat: float,
                 Vdd: float,
                 freq: float,
                 area: float,
                 size_kb: float,
                 strategies: list[ULPTechnique] = None):
        
        super().__init__("SRAM", P_dyn, P_stat, Vdd, freq, area, strategies)
        self.size_kb = size_kb

    def summary(self) -> str:
        return f"SRAM {self.size_kb}KB @ {self.freq/1e6:.1f}MHz, area={self.area}mm^2"


class PMU(HardwareModule):
    """
    Power management unit (controls DVFS, AVFS, power domains)
    """
    def __init__(self,
                 P_stat: float,
                 Vdd_domains: dict[str, float],
                 strategies: list[ULPTechnique] = None):
        
        # PMU dynamic power is usually small compared to modules
        super().__init__("PMU", P_dyn=0.0, P_stat=P_stat, Vdd=0.0, freq=0.0, area=0.0, strategies=strategies)
        self.Vdd_domains = Vdd_domains

    def summary(self) -> str:
        domains = ",".join(f"{k}:{v:.2f}V" for k, v in self.Vdd_domains.items())
        return f"PMU(domains={domains})"


class MinotaurChip:
    """
    Top-level MINOTAUR SoC, aggregating all modules
    """
    def __init__(self,
                 modules: list[HardwareModule]):
        self.modules = modules

    def total_base_power(self) -> float:
        return sum(m.base_power() for m in self.modules)

    def total_optimized_power(self) -> float:
        return sum(m.optimized_power() for m in self.modules)

    def report(self):
        print("=== MINOTAUR SoC Power Report ===")
        for m in self.modules:
            print(f"{m.summary():<40} | base={m.base_power():.3f}W  opt={m.optimized_power():.3f}W")
        print("----------------------------------")
        print(f"Total Base Power:     {self.total_base_power():.3f} W")
        print(f"Total Optimized Power:{self.total_optimized_power():.3f} W")

'''
EXAMPLE: doing power gating for RRAM modules

from ULPmodels import PowerGating
from minotaur_chip import RRAMArray, MinotaurChip

# 1) instantiate the power-gating strategy with your measured numbers
pg_strategy = PowerGating(
    P_dyn=0.05,       # e.g. 50 mW dynamic per bank
    P_stat=0.01,      # 10 mW leakage per bank
    frac_gated=1.0,   # assume you can gate 100% of the bank
    Vdd_nom=0.9       # 0.9 V supply
)

# 2) build your RRAM array, passing that strategy in
rram = RRAMArray(
    num_banks=12,
    P_dyn_per_bank=0.05,
    P_stat_per_bank=0.01,
    Vdd=0.9,
    freq=100e6,
    area_per_bank=0.1,
    strategies=[pg_strategy]
)

# 3) assemble the full SoC and run your report
chip = MinotaurChip(modules=[rram, /* other modules */])
chip.report()
'''