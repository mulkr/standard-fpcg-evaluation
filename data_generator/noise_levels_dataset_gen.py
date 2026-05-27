import os
from itertools import product
from data_gen import signal_gen
from tqdm import tqdm
from params import Noise_Param, PCG_Param, HR_Mod, fs, fhr_sig2, mhr_sig2, FREQ, GEN_LEN

noise_series = [
    Noise_Param(MFN=0.05, EN=0.05, WGN=0.025, mHS=0.10, LDI=1, LDI_num=60/20, LDI_len=(500,1500), LDI_frq=(80,fs/2), name="-4_3"),
    Noise_Param(MFN=0.10, EN=0.10, WGN=0.050, mHS=0.15, LDI=1, LDI_num=60/20, LDI_len=(500,1500), LDI_frq=(80,fs/2), name="-6_6"),
    Noise_Param(MFN=0.10, EN=0.10, WGN=0.050, mHS=0.35, LDI=1, LDI_num=60/20, LDI_len=(500,1500), LDI_frq=(80,fs/2), name="-8_1"),
    Noise_Param(MFN=0.10, EN=0.10, WGN=0.250, mHS=0.15, LDI=1, LDI_num=60/20, LDI_len=(500,1500), LDI_frq=(80,fs/2), name="-10_2"),
    Noise_Param(MFN=0.30, EN=0.30, WGN=0.050, mHS=0.15, LDI=1, LDI_num=60/20, LDI_len=(500,1500), LDI_frq=(80,fs/2), name="-11_2"),
    Noise_Param(MFN=0.10, EN=0.10, WGN=0.250, mHS=0.15, LDI=1, LDI_num=60/20, LDI_len=(500,1500), LDI_frq=(80,fs/2), name="-14_7"),
    Noise_Param(MFN=0.10, EN=0.10, WGN=0.250, mHS=0.35, LDI=1, LDI_num=60/20, LDI_len=(500,1500), LDI_frq=(80,fs/2), name="-15_3"),
    Noise_Param(MFN=0.10, EN=0.10, WGN=0.250, mHS=0.55, LDI=1, LDI_num=60/20, LDI_len=(500,1500), LDI_frq=(80,fs/2), name="-16_4"),
    Noise_Param(MFN=0.10, EN=0.10, WGN=0.250, mHS=0.75, LDI=1, LDI_num=60/20, LDI_len=(500,1500), LDI_frq=(80,fs/2), name="-17_7"),
    Noise_Param(MFN=0.10, EN=0.10, WGN=0.250, mHS=0.95, LDI=1, LDI_num=60/20, LDI_len=(500,1500), LDI_frq=(80,fs/2), name="-19_1"),
    Noise_Param(MFN=0.30, EN=0.30, WGN=0.250, mHS=0.75, LDI=1, LDI_num=60/20, LDI_len=(500,1500), LDI_frq=(80,fs/2), name="-21_0"),
    Noise_Param(MFN=0.30, EN=0.30, WGN=0.350, mHS=0.55, LDI=1, LDI_num=60/20, LDI_len=(500,1500), LDI_frq=(80,fs/2), name="-22_6"),
]

fetal_series = [
    PCG_Param(HR_MEAN=140, HR_SD=2,
              HR_SIG_1=5*fhr_sig2, HR_C1=0.01, HR_F1=0.1,
              HR_SIG_2=fhr_sig2,   HR_C2=0.03, HR_F2=0.25,
              HS_START=40,
              HS_F1=FREQ[34][0], HS_SD1=8.64,  HS_S1D=85, HS_S1D_d=12,
              HS_F2=FREQ[34][1], HS_SD2=17.81, HS_S2D=58, HS_S2D_d=9,
              HS_S1S2R=1.70, HS_S1S2R_d=0.71, HS_S1_amp=0.7, HS_AS1=0.12, name="34_140_2"),
    PCG_Param(HR_MEAN=135, HR_SD=3,
              HR_SIG_1=5*fhr_sig2, HR_C1=0.01, HR_F1=0.1,
              HR_SIG_2=fhr_sig2,   HR_C2=0.03, HR_F2=0.25,
              HS_START=37,
              HS_F1=FREQ[36][0], HS_SD1=8.64,  HS_S1D=85, HS_S1D_d=12,
              HS_F2=FREQ[36][1], HS_SD2=17.81, HS_S2D=58, HS_S2D_d=9,
              HS_S1S2R=1.70, HS_S1S2R_d=0.71, HS_S1_amp=0.7, HS_AS1=0.12, name="36_135_3"),
    PCG_Param(HR_MEAN=130, HR_SD=4,
              HR_SIG_1=5*fhr_sig2, HR_C1=0.01, HR_F1=0.1,
              HR_SIG_2=fhr_sig2,   HR_C2=0.03, HR_F2=0.25,
              HS_START=33,
              HS_F1=FREQ[38][0], HS_SD1=8.64,  HS_S1D=85, HS_S1D_d=12,
              HS_F2=FREQ[38][1], HS_SD2=17.81, HS_S2D=58, HS_S2D_d=9,
              HS_S1S2R=1.70, HS_S1S2R_d=0.71, HS_S1_amp=0.7, HS_AS1=0.12, name="38_130_4"),
]

mod_series = [
    None
]

maternal_series = [
    PCG_Param(HR_MEAN=72, HR_SD=2,
              HR_SIG_1=5*mhr_sig2, HR_C1=0.01, HR_F1=0.1,
              HR_SIG_2=mhr_sig2,   HR_C2=0.03, HR_F2=0.25,
              HS_START=0,
              HS_F1=14.93, HS_SD1=4.62,  HS_S1D=136, HS_S1D_d=6,
              HS_F2=22.44, HS_SD2=14.41, HS_S2D=95,  HS_S2D_d=7,
              HS_S1S2R=1.54, HS_S1S2R_d=0.13, HS_S1_amp=1, HS_AS1=0.12, name="72"),
]

data_dir = "noise_levels"
multiplier = 5

if not os.path.isdir(data_dir):
    os.mkdir(data_dir)

for (fet,mod,noi,mat) in tqdm(product(fetal_series,mod_series,noise_series,maternal_series),
                                     total=len(fetal_series)*len(mod_series)*len(noise_series)*len(maternal_series)):
    for i in range(multiplier):
        filename = f"GW{fet.name}{mod.name if mod is not None else ''}_SNR{noi.name}dB{'_noLDI' if noi.LDI==0 else ''}_m{mat.name}_{i}"
        filepath = os.path.join(data_dir,filename)
        signal_gen(filepath,fs,GEN_LEN,fet,mat,noi,mod)
