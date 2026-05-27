from dataclasses import dataclass

GEN_LEN = 5 # [min]
fs = 1000 #   [Hz]

fhr_sig2 = 0.03
mhr_sig2 = 0.03

# Table 2. [Hz]
FREQ = {
    34: (53.55,65.64),
    35: (45.44,63.37),
    36: (41.59,59.25),
    37: (39.39,57.94),
    38: (37.91,56.64),
    39: (37.52,55.99),
    40: (36.89,55.18),
}

@dataclass
class PCG_Param:
    HR_MEAN: float
    HR_SD: float
    HR_SIG_1: float
    HR_C1: float
    HR_F1: float
    HR_SIG_2: float
    HR_C2: float
    HR_F2: float

    HS_START: float
    HS_F1: float
    HS_SD1: float
    HS_S1D: float
    HS_S1D_d: float
    HS_F2: float
    HS_SD2: float
    HS_S2D: float
    HS_S2D_d: float
    HS_S1S2R: float
    HS_S1S2R_d: float
    HS_S1_amp: float
    HS_AS1: float

    name: str

@dataclass
class Noise_Param:
    MFN: float
    EN: float
    WGN: float
    mHS: float
    LDI: float

    LDI_num: float
    LDI_len: tuple
    LDI_frq: tuple

    name: str

@dataclass
class HR_Mod:
    POS: float
    LEN: float
    AMP: float

    name: str
