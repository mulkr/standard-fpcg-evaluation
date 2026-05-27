import numpy as np
import pyPCG.preprocessing as preproc
import pyPCG.io as io
import pyPCG
import scipy.fft as fft
import math
import os
import json
import sys
from tqdm import tqdm

# Based on:
# Tang H, Li T, Qiu T, et al (2016)
# Fetal heart rate monitoring from phonocardiograph signal using repetition frequency of heart sounds
# Journal of Electrical and Computer Engineering 2016:1–6
# https://doi.org/10.1155/2016/240426
#
# And original MATLAB code:
# https://github.com/tanghongdlut/signal-quality-assessment-of-heart-sound-signal

def nextpow2(x):
    return math.ceil(math.log(x, 2))

def fast_cfsd(sig,f1,f2,k):
    w = np.exp(-1j*2*np.pi*(f2-f1)/(k*sig.fs))
    a = np.exp(1j*2*np.pi*f1/sig.fs)
    x = preproc.envelope(sig).data
    x = x-np.mean(x)
    m = len(x)
    nfft = 2**nextpow2(m+k-1)
    kk = np.arange(-m,max(k,m))
    kk2 = (kk**2)/2
    ww = w**kk2
    nn = np.arange(m)
    aa = a**(-nn)
    aa = aa*ww[m+nn]
    y = x * aa
    fy = fft.fft(y,nfft)
    fv = fft.fft(1/ww[:k-1+m],nfft)
    fy = fy * fv #type: ignore
    g = fft.ifft(fy)
    g = g[m:m+k-1] * ww[m:m+k-1]
    return np.abs(g)

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

    M = 2e2 # bins in cycle freq domain

    # Expected FHR range BPM
    FHR_min = 80
    FHR_max = 210

    HR = dict()

    for filename in tqdm(os.listdir(datapath)):
        if not filename.endswith("wav"):
            continue
        sig = pyPCG.pcg_signal(*io.read_signal_file(os.path.join(datapath,filename),"wav"))

        windows = preproc.slice_signal(sig,10,0.5)
        hr = list()
        for win in windows:
            if len(win.data)<len(windows[0].data):
                continue
            z = fast_cfsd(win,FHR_min/60,FHR_max/60,int(M))
            fz = (np.arange(len(z)) * (FHR_max/60-FHR_min/60) / len(z))+FHR_min/60
            hr.append(fz[np.argmax(z)]*60)
        HR[filename] = hr
    with open(os.path.join(outpath,f"{method}_fhr.json"),"w") as outfile:
        outfile.write(json.dumps(HR))
