import json
import numpy as np
import pyPCG, pyPCG.io
import os
import scipy.signal as sgn
from scipy.io import loadmat
from tqdm import tqdm

if __name__ == "__main__":
    FHR_min = 80 #90
    FHR_max = 210 #200
    
    win_len = 10 #6 frame len (N)
    
    datapath = "C:/Users/mulkr/PhD/top50/"
    template = loadmat("s1template.mat")["templ"].T[0]
    template = template/np.max(template)
    
    HR = dict()
    sigma_r = 50
    for filename in tqdm(os.listdir(datapath)):
        if not filename.endswith("wav"):
            continue
        sig = pyPCG.pcg_signal(*pyPCG.io.read_signal_file(os.path.join(datapath,filename),"wav"))
        sig = pyPCG.normalize(sig)
        matched = sgn.correlate(sig.data,template,mode="same")
        energy = np.square(matched[1:-1]) - (matched[:-2]*matched[2:])
        energy = energy/np.max(energy)
        
        at_end = False
        win_len_ind = win_len*sig.fs
        step = win_len_ind//2
        start = 0
        fhr_min_ind = (sig.fs*60)/FHR_min
        fhr_max_ind = (sig.fs*60)/FHR_max
        FHR = []
        # MERIT = []
        while not at_end:
            end = start+win_len_ind
            if end >= len(energy):
                end = len(energy)-1
                at_end = True
            win = energy[start:end]
            acorr = sgn.correlate(win,win,mode="full")
            acorr = acorr[len(acorr)//2:]
            peaks,_ = sgn.find_peaks(acorr)
            peaks = peaks[peaks>fhr_max_ind]
            peaks = peaks[peaks<fhr_min_ind]
            
            R_a0 = acorr[0]
            
            FHR.append((sig.fs*60)/peaks[np.argmax(acorr[peaks])])
            start += step
            
        FHR = np.array(FHR)

        HR[filename] = FHR.tolist()
    with open("zahorian.json","w") as outfile:
        outfile.write(json.dumps(HR))
