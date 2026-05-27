import numpy as np
import pyPCG, pyPCG.io, pyPCG.preprocessing
import os
import sys
import scipy.signal as sgn
from tqdm import tqdm

# Based on:
# Ruffo M, Cesarelli M, Romano M, et al (2010)
# An algorithm for fhr estimation from foetal phonocardiographic signals
# Biomedical Signal Processing and Control 5(2):131–141
# https://doi.org/10.1016/j.bspc.2010.02.00
#
# And Appendix A of:
# Cesarelli M, Ruffo M, Romano M, et al (2012)
# Simulation of foetal phonocardiographic recordings for testing of fhr extraction algorithms
# Computer Methods and Programs in Biomedicine 107(3):513–523
# https://doi.org/10.1016/j.cmpb.2011.11.008

if __name__ == "__main__":
    if len(sys.argv)<3:
        print("Please specify an input and output location")
        print("<method>.py <input dir> <output dir>")
        exit()

    method = sys.argv[0][:-3]
    datapath = sys.argv[1]
    outpath = sys.argv[2]

    if not os.path.isdir(outpath):
        os.mkdir(outpath)

    if not os.path.isdir(os.path.join(outpath,method)):
        os.mkdir(os.path.join(outpath,method))

    # Used to initialize the method
    training_window_len = 5 # seconds, Originally: 5

    # Interval to search for candidate S1
    candidate_lo_th = 0.64 # a.u., Originally: 0.65
    candidate_hi_th = 1.35 # a.u.

    for filename in tqdm(os.listdir(datapath)):
        if not filename.endswith("wav"):
            continue
        sig = pyPCG.pcg_signal(*pyPCG.io.read_signal_file(os.path.join(datapath,filename),"wav"))
        sig = pyPCG.normalize(sig)

        # B2.2 (Ruffo 2010)
        lp = pyPCG.preprocessing.filter(sig, 4, 54)
        hp = pyPCG.preprocessing.filter(lp, 4, 34, "HP")
        data = np.array(hp.data)
        fs = hp.fs

        # B3.1 (Ruffo 2010)
        energy = np.square(data[1:-1]) - (data[:-2]*data[2:])
        energy = energy/np.max(energy)

        # B4 (Ruffo 2010)
        train_win = energy[:fs*training_window_len]
        peaks, _ = sgn.find_peaks(train_win,distance=fs//3)
        at_end = False
        T0 = peaks[-1]
        HT_LT = 0.5
        while not at_end:
            wpeaks = peaks[-8:]
            TH = np.mean(energy[wpeaks])*HT_LT
            MEAN = np.mean(np.diff(wpeaks))
            start =round(T0+candidate_lo_th*MEAN)
            end = round(T0+candidate_hi_th*MEAN)
            if end>=len(energy):
                at_end = True
                end = len(energy)-1
            if start >= len(energy):
                break

            win = energy[start:end]
            winpeaks,_ = sgn.find_peaks(win,height=TH)
            if len(winpeaks)>0:
                if len(winpeaks)>1:
                    detect = winpeaks[np.argmin(np.abs(np.abs(winpeaks+start-T0)-MEAN))]
                else:
                    detect = winpeaks
                if HT_LT == 0.3:
                    HT_LT = 0.5
            else:
                if HT_LT == 0.3:
                    T0 += end
                    continue
                HT_LT = 0.3
                continue
            P = start+detect
            if P>=len(energy):
                break
            peaks = np.append(peaks, start+detect)
            T0 = peaks[-1]

        with open(os.path.join(outpath,method,f"{filename[:-4]}.csv"),"w") as out_file:
            out_file.write("Location;Value\n")
            for s in peaks:
                out_file.write(f"{s/fs};S1\n")


