import json
import numpy as np
import pyPCG, pyPCG.io, pyPCG.preprocessing
import os
import scipy.signal as sgn
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view

if __name__ == "__main__":
    FHR_max = 210 #200
    FHR_min = 80 #22 - 100
    datapath = "C:/Users/mulkr/PhD/top50/"
    
    # IMPORTANT!
    # This parameter is not specified in the article
    RMS_size = 0.200
    
    HR = dict()
    for filename in tqdm(os.listdir(datapath)):
        if not filename.endswith("wav"):
            continue
        sig = pyPCG.pcg_signal(*pyPCG.io.read_signal_file(os.path.join(datapath,filename),"wav"))
        sig = pyPCG.normalize(sig)
        sig = pyPCG.preprocessing.filter(sig, 4, 24,"HP") # optional
        sig = pyPCG.preprocessing.filter(sig, 4, 54) # optional
        pcg = sig.data
        data = sig.data
        fs = sig.fs
        
        L = round(RMS_size*fs)
        A = np.sqrt(np.mean(np.square(sliding_window_view(data,L)),axis=1))
        
        # IMPORTANT!
        # This was added by us. Otherwise it simply does not give the expected output
        A = (A-np.min(A))
        A = A/np.max(A)
        A_thd = np.median(A)
        
        peaks, _ = sgn.find_peaks(A,height=A_thd)
        new_peaks = []
        
        for i,p in enumerate(peaks):
            if i==0:
                new_peaks.append(p)
                continue
            prev = new_peaks[-1]
            # peak fusion
            new_peak = p
            if prev >= len(A):
                continue
            if (p-prev)<(0.030*fs):
                new_peaks.pop()
                new_peak = round((A[p]*p + A[prev]*prev)/(A[p]+A[prev]))
            new_peaks.append(new_peak)
        peaks = new_peaks
        new_peaks = []
        for i,p in enumerate(peaks):
            if i == 0:
                new_peaks.append(p)
                continue
            prev = new_peaks[-1]
            # amplitude regularity
            if not (A[prev]*2.5)>A[p]:
                continue
            new_peaks.append(p)

        peaks = new_peaks
        new_peaks = []
        for i,p in enumerate(peaks):
            if i == 0:
                new_peaks.append(p)
                continue
            prev = new_peaks[-1]
            # FHR range
            q = p-prev
            if (60*fs)/FHR_max > q:
                continue
            new_peaks.append(p)

        weighted,weights = [],[]
        for i,p in enumerate(new_peaks):
            if i == 0:
                continue
            prev = new_peaks[i-1]
            weights.append(A[p]+A[prev])
            weighted.append((p-prev)*(A[p]+A[prev]))
        
        # IMPORTANT!
        # This is probably not correct. The article does not describe properly if these should be calculated by time or "by detection"
        # Instead of the FHR calculated this way, we used our FHR calculation based on the detections
        # See: run_bechmark.py
        win_len = 10
        overlap = 0.5
        step = win_len-(win_len*overlap)
        win_len = round(win_len*sig.fs)
        step = round(step*sig.fs)
        at_end = False
        start = 0
        s = np.array(new_peaks)
        hr = list()
        while not at_end:
            end = start+win_len
            if end >= s[-1]:
                end = s[-1]
                at_end = True
            subwin = s[s>start]
            subwin = subwin[subwin<end]
            w_sum = np.sum(A[subwin])
            s1diff = np.diff(subwin)
            s1diff = np.append([0],s1diff)
            hr.append(np.sum(s1diff*A[subwin])/w_sum)
            start += step
        hr = np.array(hr)
        FHR = 60/(hr/fs)
        HR[filename] = FHR.tolist()
        
        S1 = new_peaks
        
        with open(f"chen/{filename[:-4]}.csv","w") as out_file:
            out_file.write("Location;Value\n")
            for s in S1:
                out_file.write(f"{s/fs+RMS_size/2};S1\n")
    with open("chen_fhr.json","w") as outfile:
        outfile.write(json.dumps(HR))
