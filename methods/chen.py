import json
import numpy as np
import pyPCG, pyPCG.io, pyPCG.preprocessing
import os
import sys
import scipy.signal as sgn
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view

# Based on:
# Chen J, Phua K, Song Y, et al (2006)
# Fetal heart signal monitoring with confidence factor
# In: 2006 IEEE International Conference on Multimedia and Expo, pp 1937–1940
# https://doi.org/10.1109/ICME.2006.262936

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

    # Maximum and minimum expected FHR BPM
    FHR_max = 210 # Originally: 200
    FHR_min = 80  # Originally: 100

    # Amplitude regularity factor
    gamma = 2.5 # Originally: 2

    # Peak fusion window size
    fusion_win = 0.030 # seconds

    # IMPORTANT!
    # This parameter is not specified in the article
    RMS_size = 0.200 # seconds

    HR = dict()
    for filename in tqdm(os.listdir(datapath)):
        if not filename.endswith("wav"):
            continue
        sig = pyPCG.pcg_signal(*pyPCG.io.read_signal_file(os.path.join(datapath,filename),"wav"))
        sig = pyPCG.normalize(sig)
        sig = pyPCG.preprocessing.filter(sig, 4, 24,"HP") # Mentioned as optional
        sig = pyPCG.preprocessing.filter(sig, 4, 54) # Mentioned as optional
        pcg = sig.data
        data = sig.data
        fs = sig.fs

        # Envelope calculation
        L = round(RMS_size*fs)
        A = np.sqrt(np.mean(np.square(sliding_window_view(data,L)),axis=1))

        # IMPORTANT!
        # This was added by us. Otherwise it simply does not give the expected output
        A = (A-np.min(A))
        A = A/np.max(A)
        A_thd = np.median(A)

        # Find peaks as in Eq.(2)
        peaks, _ = sgn.find_peaks(A,height=A_thd)

        # Peak fusion
        new_peaks = []
        for i,p in enumerate(peaks):
            if i==0:
                new_peaks.append(p)
                continue
            prev = new_peaks[-1]
            new_peak = p
            if prev >= len(A):
                continue
            if (p-prev)<(fusion_win*fs):
                new_peaks.pop()
                new_peak = round((A[p]*p + A[prev]*prev)/(A[p]+A[prev]))
            new_peaks.append(new_peak)
        peaks = new_peaks

        # Amplitude regularity filtering
        new_peaks = []
        for i,p in enumerate(peaks):
            if i == 0:
                new_peaks.append(p)
                continue
            prev = new_peaks[-1]
            if not (A[prev]*gamma)>A[p]:
                continue
            new_peaks.append(p)
        peaks = new_peaks

        # FHR range based filtering
        new_peaks = []
        for i,p in enumerate(peaks):
            if i == 0:
                new_peaks.append(p)
                continue
            prev = new_peaks[-1]
            q = p-prev
            if (60*fs)/FHR_max > q:
                continue
            new_peaks.append(p)

        # Calculating peak weights based on (5)
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

        with open(os.path.join(outpath,method,f"{filename[:-4]}.csv"),"w") as out_file:
            out_file.write("Location;Value\n")
            for s in S1:
                out_file.write(f"{s/fs+RMS_size/2};S1\n")
    with open(os.path.join(outpath,f"{method}_fhr.json"),"w") as outfile:
        outfile.write(json.dumps(HR))
