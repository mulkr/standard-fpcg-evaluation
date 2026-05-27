import numpy as np
import scipy.fft as fft
import scipy.signal as signal
import scipy.io.wavfile as wav
import csv
from params import PCG_Param, Noise_Param, HR_Mod

# seed rng for reproducibility
rng = np.random.default_rng(100)

def HR_gen(hr_mean, hr_sd, gen_len, fs, hr_sig1, hr_c1, hr_f1, hr_sig2, hr_c2, hr_f2):

    f = np.linspace(0, (hr_mean/60)/2, 2000) # N into parameter?

    # R-R spectrum model
    mayer = hr_sig1 / np.sqrt(2 * np.pi * hr_c1**2) * np.exp(-((f - hr_f1) ** 2) / (2 * hr_c1**2))
    rsa = hr_sig2 / np.sqrt(2 * np.pi * hr_c2**2) * np.exp(-((f - hr_f2) ** 2) / (2 * hr_c2**2))
    S = mayer + rsa

    # build complex spectrum from model
    F_m = np.sqrt(S)
    F_a = rng.random(F_m.shape) * 2 * np.pi
    F_pos = F_m * np.exp(1j * F_a)

    # make it symmetric
    N = len(F_pos)
    F_full = np.zeros(N * 2, dtype=complex)
    F_full[0] = F_pos[0].real
    F_full[1:N] = F_pos[1:N]
    F_full[N] = 0
    F_full[N + 1 :] = np.conj(F_pos[1:N][::-1])

    # timeseries from spectrum
    RR_mean = 60 / hr_mean
    T = fft.ifft(F_full).real + RR_mean #type:ignore

    # R-R to HR
    t_hr = np.cumsum(T)
    y_hr = 60 / T
    fhr_t = np.arange(0, t_hr[-1], 1/fs)
    fhr_interp = np.interp(fhr_t, t_hr, y_hr)

    HR = (fhr_interp-np.mean(fhr_interp))/np.std(fhr_interp) * hr_sd + hr_mean

    # cut it to size
    HR = HR[:gen_len*60*fs]
    t = fhr_t[:gen_len*60*fs]

    return HR, t

def HS_gen(hr, t, start, gen_len, fs, f1, sd1, s1d, s1d_d, f2, sd2, s2d, s2d_d, s1s2r, s1s2r_d, s1a, as1, fetal):

    # HS timing parameters based on FHR
    B2Bs = 60000/hr # [ms]
    SSIDs = 210 - 0.5 * hr # [ms]
    if not fetal:
        SSIDs = 0.2*(60000/hr) + 160

    # S1 positions
    curr = start
    S1p = [curr]
    while curr<gen_len*60*fs:
        S1p.append(S1p[-1]+B2Bs[int(curr)])
        curr += B2Bs[int(curr)]

    # S1 and S2 signals
    S1s = np.zeros_like(t)
    S2s = np.zeros_like(t)

    # generate gaussian pulses for S1 and S2 templates
    pf1 = f1# rng.normal(f1,np.sqrt(sd1)/2)
    pw1 = abs(2000/(rng.normal(s1d,np.sqrt(s1d_d)/2)*f1))
    if not fetal:
        pw1 /= 2

    pf2 = f2# rng.normal(f2,np.sqrt(sd2)/2)
    pw2 = abs(2000/(rng.normal(s2d,np.sqrt(s2d_d)/2)*f2))
    if not fetal:
        pw2 /= 2

    tc: float = signal.gausspulse("cutoff",abs(pf1),pw1,tpr=-80) #type:ignore
    gt = np.arange(-tc,tc,1/fs)
    S1_puls = signal.gausspulse(gt,abs(pf1),pw1) #type:ignore

    tc = signal.gausspulse("cutoff",abs(pf2),pw2,tpr=-80) #type:ignore
    gt = np.arange(-tc,tc,1/fs)
    S2_puls = signal.gausspulse(gt,abs(pf2),pw2) #type:ignore

    s1s2r = rng.normal(s1s2r,np.sqrt(s1s2r_d)/2)

    S2p = []
    # create HS signals
    for s1_pos,ssid in zip(S1p,SSIDs):
        s1_amp = rng.normal(s1a,as1)
        S1s[min(int(s1_pos),len(S1s)-1)] = s1_amp

        s2_pos = s1_pos + ssid
        S2p.append(s2_pos)
        s2_amp = s1_amp / s1s2r
        S2s[min(int(s2_pos),len(S2s)-1)] = s2_amp

    S1s = signal.convolve(S1s,S1_puls,"same")
    S2s = signal.convolve(S2s,S2_puls,"same")

    sig = S1s + S2s
    return sig, S1p, S2p


def signal_gen(filename: str, fs: float, gen_len: float, fetal_params: PCG_Param, maternal_params: PCG_Param, noise_params: Noise_Param, fhr_mod: HR_Mod|None) -> None:
    FHR, t = HR_gen(fetal_params.HR_MEAN, fetal_params.HR_SD, gen_len, fs,
                    fetal_params.HR_SIG_1, fetal_params.HR_C1, fetal_params.HR_F1,
                    fetal_params.HR_SIG_2, fetal_params.HR_C2, fetal_params.HR_F2)

    # accel-decel
    if fhr_mod is not None:
        mod = np.exp(-((t - fhr_mod.POS) ** 2) / (fhr_mod.LEN**2))
        mod = mod/np.max(mod) * fhr_mod.AMP
        FHR += mod

    # generate fpcg
    fPCG, S1_loc, S2_loc = HS_gen(FHR, t, fetal_params.HS_START, gen_len, fs,
                                  fetal_params.HS_F1, fetal_params.HS_SD1, fetal_params.HS_S1D, fetal_params.HS_S1D_d,
                                  fetal_params.HS_F2, fetal_params.HS_SD2, fetal_params.HS_S2D, fetal_params.HS_S2D_d,
                                  fetal_params.HS_S1S2R, fetal_params.HS_S1S2R_d,
                                  fetal_params.HS_S1_amp, fetal_params.HS_AS1, fetal=True)

    # add noise
    b,a = signal.butter(5,25,"lowpass",fs=fs) #type:ignore
    MFN = signal.filtfilt(b,a,rng.standard_normal(len(fPCG))) * noise_params.MFN

    b,a = signal.butter(5,100,"highpass",fs=fs) #type:ignore
    EN = signal.filtfilt(b,a,rng.standard_normal(len(fPCG))) * noise_params.EN

    WGN = rng.standard_normal(len(fPCG)) * noise_params.WGN

    ldi_num = int(np.round(rng.normal(gen_len*noise_params.LDI_num,1)))
    ldi_lens = rng.integers(noise_params.LDI_len[0],noise_params.LDI_len[1],ldi_num)
    ldi_freq = rng.uniform(noise_params.LDI_frq[0],noise_params.LDI_frq[1],ldi_num)
    ldi_pos = rng.integers(0,len(fPCG)-1,ldi_num)

    LDI = np.zeros_like(fPCG)
    for lg,p,f in zip(ldi_lens,ldi_pos,ldi_freq):
        win = t[p:p+lg]
        LDI[p:p+lg] += np.sin(2*np.pi*win*f)
    LDI = LDI * noise_params.LDI

    mHR, t = HR_gen(maternal_params.HR_MEAN, maternal_params.HR_SD, gen_len, fs,
                    maternal_params.HR_SIG_1, maternal_params.HR_C1, maternal_params.HR_F1,
                    maternal_params.HR_SIG_2, maternal_params.HR_C2, maternal_params.HR_F2)
    mPCG,_,_ = HS_gen(mHR, t, maternal_params.HS_START, gen_len, fs,
                      maternal_params.HS_F1, maternal_params.HS_SD1, maternal_params.HS_S1D, maternal_params.HS_S1D_d,
                      maternal_params.HS_F2, maternal_params.HS_SD2, maternal_params.HS_S2D, maternal_params.HS_S2D_d,
                      maternal_params.HS_S1S2R, maternal_params.HS_S1S2R_d,
                      maternal_params.HS_S1_amp, maternal_params.HS_AS1, fetal=False)
    mHS = mPCG * noise_params.mHS

    sig =  fPCG + mHS + MFN + EN + WGN + LDI

    dtype = np.int16
    sig[sig>1] = 1
    sig[sig<-1] = -1
    sig *= np.iinfo(dtype).max

    wav.write(f"{filename}.wav",fs,sig.astype(dtype))

    with open(f"{filename}.csv", 'w') as labelfile:
        writer = csv.writer(labelfile,delimiter=';')
        writer.writerow(["Location","Value"])
        for s1 in S1_loc:
            writer.writerow([s1/fs,"S1"])
        for s2 in S2_loc:
            writer.writerow([s2/fs,"S2"])
