import numpy as np
import pyPCG, pyPCG.io, pyPCG.preprocessing
import os
import scipy.signal as sgn
from tqdm import tqdm


if __name__ == "__main__":
    datapath = "C:/Users/mulkr/PhD/top50/"
    for filename in tqdm(os.listdir(datapath)):
        if not filename.endswith("wav"):
            continue
        sig = pyPCG.pcg_signal(*pyPCG.io.read_signal_file(os.path.join(datapath,filename),"wav"))
        sig = pyPCG.normalize(sig)
        lp = pyPCG.preprocessing.filter(sig, 4, 54)
        hp = pyPCG.preprocessing.filter(lp, 4, 34,"HP")

        data = np.array(hp.data)
        energy = np.square(data[1:-1]) - (data[:-2]*data[2:])
        energy = energy/np.max(energy)

        fs = hp.fs
        
        train_win = energy[:fs*4]
        peaks, _ = sgn.find_peaks(train_win,distance=fs//3)
        at_end = False
        T0 = peaks[-1]
        while not at_end:
            wpeaks = peaks[-8:]
            TH = np.mean(energy[wpeaks])/2
            MEAN = np.mean(np.diff(wpeaks))
            start =round(T0+0.64*MEAN) 
            end = round(T0+1.35*MEAN)
            if end>=len(energy):
                at_end = True
                end = len(energy)-1
            if start >= len(energy):
                at_end = True
                break
                
            win = energy[start:end]
            winpeaks,_ = sgn.find_peaks(win,height=TH)
            if len(winpeaks)>0:
                if len(winpeaks)>1:
                    detect = winpeaks[np.argmin(np.abs(np.abs(winpeaks+start-T0)-MEAN))]
                else:
                    detect = winpeaks
            P = start+detect
            if P>=len(energy):
                break
            peaks = np.append(peaks, start+detect)
            T0 = peaks[-1]
        with open(f"cesarelli/{filename[:-4]}.csv","w") as out_file:
            out_file.write("Location;Value\n")
            for s in peaks:
                out_file.write(f"{s/fs};S1\n")
        

