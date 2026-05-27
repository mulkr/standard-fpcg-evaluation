import os
import datetime
import numpy as np
import pyPCG, pyPCG.lr_hsmm, pyPCG.segment, pyPCG.io, pyPCG.preprocessing
from joblib import Parallel, delayed
import sys
from tqdm_joblib import tqdm_joblib
from tqdm import tqdm

JOBS = 5

SIG_FS = 333
F_FS = 50
BP = (15,55)
S1 = 70
S2 = 65
HR = (100,200)
S1_STD = 22
S2_STD = 22

def align_timing(start,end):
    a_start, a_end = start,end
    if start[0] > end[0]:
        a_end = end[1:]
    if start[-1] > end[-1]:
        a_start = start[:-1]
    return a_start, a_end

if __name__ == "__main__":
    if len(sys.argv)<4:
        print("Please specify training data, testing data and output directories")
        print("<method>.py <train dir> <test dir> <output dir>")
        exit()

    method = sys.argv[0][:-3]
    datapath = sys.argv[1]
    labelpath = datapath
    testpath = sys.argv[2]
    outpath = sys.argv[3]
    if len(sys.argv)>4:
        labelpath = sys.argv[4]

    if not os.path.isdir(outpath):
        os.mkdir(outpath)

    if not os.path.isdir(os.path.join(outpath,method)):
        os.mkdir(os.path.join(outpath,method))

    print("======================================")
    print(f"Training and testing {pyPCG.__version__} - Processing jobs: {JOBS}")
    print(f"Started: {datetime.datetime.now().time()}")

    print("Reading training data...")

    signals = []
    s1_labels,s2_labels = [],[]
    filenames = []

    for filename in tqdm(os.listdir(datapath)):
        if not filename.endswith("wav"):
            continue
        data, fs = pyPCG.io.read_signal_file(os.path.join(datapath,filename),"wav")
        label = pyPCG.io.read_hsannot_file(os.path.join(labelpath,f"{filename[:-4]}.csv"))
        filenames.append(filename)
        raw = pyPCG.pcg_signal(data,fs)
        signal = pyPCG.normalize(raw)
        signal = pyPCG.preprocessing.wt_denoise_sth(signal)
        signal = pyPCG.preprocessing.resample(signal, SIG_FS)
        signals.append(signal.data)
        s1_labels.append(label[0])
        s2_labels.append(label[1])

    signals = np.array(signals)
    s1_labels = np.array(s1_labels,dtype=object)
    s2_labels = np.array(s2_labels,dtype=object)
    filenames = np.array(filenames)

    print("Reading training done!")

    def calc_patient(singal_data):
        features = pyPCG.lr_hsmm._generate_features(singal_data,SIG_FS,F_FS,preproc=BP)
        return features

    print("Preprocessing training...")
    with tqdm_joblib(total=len(signals)):
        tot_features = Parallel(n_jobs=JOBS)(delayed(calc_patient)(patients) for patients in signals)
        tot_features = np.array(tot_features,dtype=object)
    print("Preprocessing done!")

    print("Training...")
    flat_features = [[],[],[],[]]
    for record in tot_features:
        for ind,feature in enumerate(record):
            flat_features[ind] = flat_features[ind]+feature.tolist()

    flat_features = np.array(flat_features)

    model = pyPCG.lr_hsmm.LR_HSMM()
    model.signal_fs = SIG_FS
    model.feature_fs = F_FS
    model.bandpass_frq = BP
    model.expected_hr_range = HR
    model.mean_s1_len = S1
    model.mean_s2_len = S2
    model.std_s1_len = S1_STD
    model.std_s2_len = S2_STD
    model.train_with_precalc_features(flat_features,signals,s1_labels,s2_labels) #type: ignore
    model.save_model(os.path.join(outpath,"muller_trained_complete.json"))
    print("Training done!")

    print("Testing...")
    print("Reading testing data...")
    test_signals = []
    test_fnames = []
    for filename in tqdm(os.listdir(testpath)):
        if not filename.endswith("wav"):
            continue
        data, fs = pyPCG.io.read_signal_file(os.path.join(testpath,filename),"wav")
        test_fnames.append(filename)
        raw = pyPCG.pcg_signal(data,fs)
        signal = pyPCG.normalize(raw)
        signal = pyPCG.preprocessing.wt_denoise_sth(signal)
        test_signals.append(signal)
    print("Reading testing done!")

    print("Testing...")
    s1_pred_out,s2_pred_out,fnames_out = [],[],[]
    for t_signal,t_file in tqdm(zip(test_signals,test_fnames),total=len(test_signals)):
        states = pyPCG.segment.segment_hsmm(model,t_signal,recalc=True)
        pred_s1s,pred_s1e = pyPCG.segment.convert_hsmm_states(states,pyPCG.segment.heart_state.S1)
        pred_s2s,pred_s2e = pyPCG.segment.convert_hsmm_states(states,pyPCG.segment.heart_state.S2)

        pred_s1s, pred_s1e = align_timing(pred_s1s,pred_s1e)
        pred_s2s, pred_s2e = align_timing(pred_s2s,pred_s2e)
        pred_s1 = (pred_s1s+pred_s1e)/2
        pred_s2 = (pred_s2s+pred_s2e)/2

        s1_pred_out.append(pred_s1/model.signal_fs)
        s2_pred_out.append(pred_s2/model.signal_fs)
        fnames_out.append(t_file)
    print("Testing  done!")

    for fname,s1_pred,s2_pred in zip(fnames_out,s1_pred_out,s2_pred_out):
        with open(os.path.join(outpath,method,f"{fname[:-4]}.csv"),"w") as out_file:
            out_file.write("Location;Value\n")
            for s in s1_pred:
                out_file.write(f"{s};S1\n")
            for s in s2_pred:
                out_file.write(f"{s};S2\n")
    print(f"Ended: {datetime.datetime.now().time()}")
    print("======================================")
