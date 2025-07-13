import numpy as np
import pyPCG.preprocessing as preproc
import pyPCG.io as io
import pyPCG
import scipy.fft as fft
import math
import os
import json

def nextpow2(x):
    return math.ceil(math.log(x, 2))

def fast_cfsd(sig,f1,f2,k):
    # from Tang et al.
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
    
    M = 2e2 # bins in cycle freq domain
    
    # BPM range: 80-210
    f_start = 1.33
    f_end = 3.5
    
    HR = dict()
    
    datapath = "C:/Users/mulkr/PhD/top50/"
    for filename in os.listdir(datapath):
        if not filename.endswith("wav"):
            continue
        sig = pyPCG.pcg_signal(*io.read_signal_file(os.path.join(datapath,filename),"wav"))
        
        windows = preproc.slice_signal(sig,10,0.5)
        hr = list()
        for win in windows:
            if len(win.data)<len(windows[0].data):
                continue
            z = fast_cfsd(win,f_start,f_end,int(M))
            fz = (np.arange(len(z)) * (f_end-f_start) / len(z))+f_start
            hr.append(fz[np.argmax(z)]*60)
        HR[filename] = hr
    with open("tang.json","w") as outfile:
        outfile.write(json.dumps(HR))