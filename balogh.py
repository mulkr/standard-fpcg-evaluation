import numpy as np
import pyPCG, pyPCG.io, pyPCG.preprocessing
import os
import scipy.signal as sgn
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view

def method(data,N1,N2):
    t1 = sliding_window_view(data,N1)
    t2 = sliding_window_view(np.roll(data,-N1),N1)
    I1 = (np.sum(t2,axis=1)-np.sum(t1,axis=1))
    I1 = np.append(np.zeros(N1-1),I1)
    
    temp = data+I1
    t3 = sliding_window_view(temp,N2)
    t4 = sliding_window_view(np.roll(temp,-N2),N2)
    I2 = (np.sum(t4,axis=1)-np.sum(t3,axis=1))
    
    t5 = sliding_window_view(I2,N2)
    t6 = sliding_window_view(np.roll(I2,-N2),N2)
    V = -(np.sum(t6,axis=1)-np.sum(t5,axis=1))
    V[V<=0]=0
    V = V/np.max(V)
    V = np.append(np.zeros(2*N2+2*N1),V)
    return V

if __name__ == "__main__":
    datapath = "C:/Users/mulkr/PhD/top50/"
    for filename in tqdm(os.listdir(datapath)):
        if not filename.endswith("wav"):
            continue
        sig = pyPCG.pcg_signal(*pyPCG.io.read_signal_file(os.path.join(datapath,filename),"wav"))
        sig = pyPCG.normalize(sig)
        pcg = sig.data
        fs = sig.fs
        sig = pyPCG.preprocessing.filter(sig, 4, 24,"HP") # optional
        
        # IMPORTANT!
        # This was added by us. Otherwise it simply does not give the expected output
        # Squaring optional
        sig = pyPCG.preprocessing.envelope(sig)
        # data = np.square(sig.data)
        data = sig.data
        
        N1 = round(0.022*sig.fs)
        N2 = round(0.220*sig.fs)
        
        V_cycle = method(data,N1,N2)
        
        N1 = round(0.011*sig.fs)
        N2 = round(0.110*sig.fs)
        V_hs = method(data,N1,N2)
        
        S1, _ = sgn.find_peaks(V_cycle,distance=N2)
        
        S1_n = np.append(S1[1:],len(V_hs)-1)
        S2 = []
        for s1,s1n in zip(S1,S1_n):
            peaks, _= sgn.find_peaks(V_hs[s1:s1n])
            if len(peaks)==0:
                continue
            mid = (s1n-s1)/2
            s2 = peaks[np.argmin(np.abs(peaks-mid))]+s1
            S2.append(s2)
        
        S2 = np.array(S2)

        with open(f"balogh/{filename[:-4]}.csv","w") as out_file:
            out_file.write("Location;Value\n")
            for s in S1:
                out_file.write(f"{s/fs};S1\n")
            for s in S2:
                out_file.write(f"{s/fs};S2\n")