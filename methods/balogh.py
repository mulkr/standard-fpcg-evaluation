import numpy as np
import pyPCG, pyPCG.io, pyPCG.preprocessing
import os
import sys
import scipy.signal as sgn
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view

# Based on:
# Balogh ÁT, Kovács F (2011)
# Application of phonocardiography on preterm infants with patent ductus arteriosus
# Biomedical Signal Processing and Control 6(4):337–345
# https://doi.org/10.1016/j.bspc.2011.05.009
# And:
# Kósa E, Balogh ÁT, Üveges B, et al (2008)
# Heuristic method for heartbeat detection in fetal phonocardiographic signals
# In: 2008 International Conference on Signals and Electronic Systems, pp 231–234
# https://doi.org/10.1109/ICSES.2008.4673401

def calc_method(data,N1,N2):
    # Eq.(1) (Balogh 2011)
    term_1 = sliding_window_view(data,N1)
    term_2 = sliding_window_view(np.roll(data,-N1),N1)
    I1 = (np.sum(term_2,axis=1)-np.sum(term_1,axis=1))
    I1 = np.append(np.zeros(N1-1),I1)

    # Eq.(2) (Balogh 2011)
    temp = data+I1
    term_1 = sliding_window_view(temp,N2)
    term_2 = sliding_window_view(np.roll(temp,-N2),N2)
    I2 = (np.sum(term_2,axis=1)-np.sum(term_1,axis=1))

    # Eq.(3) (Balogh 2011)
    term_1 = sliding_window_view(I2,N2)
    term_2 = sliding_window_view(np.roll(I2,-N2),N2)
    V = -(np.sum(term_2,axis=1)-np.sum(term_1,axis=1))
    V[V<=0]=0
    V = V/np.max(V)
    V = np.append(np.zeros(2*N2+2*N1),V)
    return V

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

    # Fig.2. (Balogh 2011)
    # Heart sounds
    HS_N1 = 0.011 # seconds
    HS_N2 = 0.110 # seconds
    # Heart cycles
    CY_N1 = 0.022 # seconds
    CY_N2 = 0.220 # seconds

    for filename in tqdm(os.listdir(datapath)):
        if not filename.endswith("wav"):
            continue
        sig = pyPCG.pcg_signal(*pyPCG.io.read_signal_file(os.path.join(datapath,filename),"wav"))
        sig = pyPCG.normalize(sig)
        pcg = sig.data
        fs = sig.fs
        sig = pyPCG.preprocessing.filter(sig, 4, 24,"HP") # Mentioned as optional

        # IMPORTANT!
        # This was added by us. Otherwise it simply does not give the expected output
        sig = pyPCG.preprocessing.envelope(sig)
        # Squaring optional
        # data = np.square(sig.data)
        data = sig.data

        # Heart cycle specific calc
        N1 = round(CY_N1*sig.fs)
        N2 = round(CY_N2*sig.fs)
        V_cycle = calc_method(data,N1,N2)

        # Heart sound specific calc
        N1 = round(HS_N1*sig.fs)
        N2 = round(HS_N2*sig.fs)
        V_hs = calc_method(data,N1,N2)

        # Every heart cycle gives an S1
        S1, _ = sgn.find_peaks(V_cycle,distance=N2)

        # Find S2 based on S1
        S1_next = np.append(S1[1:],len(V_hs)-1)
        S2 = []
        for s1,s1n in zip(S1,S1_next):
            peaks, _= sgn.find_peaks(V_hs[s1:s1n])
            if len(peaks)==0:
                continue
            mid = (s1n-s1)/2
            s2 = peaks[np.argmin(np.abs(peaks-mid))]+s1
            S2.append(s2)

        S2 = np.array(S2)

        with open(os.path.join(outpath,method,f"{filename[:-4]}.csv"),"w") as out_file:
            out_file.write("Location;Value\n")
            for s in S1:
                out_file.write(f"{s/fs};S1\n")
            for s in S2:
                out_file.write(f"{s/fs};S2\n")
