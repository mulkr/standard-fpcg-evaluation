import os
import datetime
import numpy as np
import pyPCG, pyPCG.lr_hsmm, pyPCG.segment, pyPCG.io, pyPCG.preprocessing
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
import sys

CROSS_VALIDATION_FOLD = 10
JOBS = 5

SIG_FS = 1000
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
    if len(sys.argv)<3:
        print("Please specify an input and output location")
        print("<method>.py <input dir> <output dir>")
        exit()

    method = sys.argv[0][:-3]
    datapath = sys.argv[1]
    outpath = sys.argv[2]
    labelpath = datapath
    if len(sys.argv)>3:
        labelpath = sys.argv[3]

    if not os.path.isdir(outpath):
        os.mkdir(outpath)

    if not os.path.isdir(os.path.join(outpath,method)):
        os.mkdir(os.path.join(outpath,method))

    print("======================================")
    print(f"Cross validation {pyPCG.__version__} - Folds: {CROSS_VALIDATION_FOLD} - Processing jobs: {JOBS}")
    print(f"Started: {datetime.datetime.now().time()}")

    print("Reading patient data...")

    signals = []
    s1_labels,s2_labels = [],[]
    filenames = []

    for filename in os.listdir(datapath):
        if not filename.endswith("wav"):
            continue
        data, fs = pyPCG.io.read_signal_file(os.path.join(datapath,filename),"wav")
        label = pyPCG.io.read_hsannot_file(os.path.join(labelpath,f"{filename[:-4]}.csv"))
        print(filename)
        filenames.append(filename)
        raw = pyPCG.pcg_signal(data,fs)
        signal = pyPCG.normalize(raw)
        signal = pyPCG.preprocessing.wt_denoise_sth(signal)
        signals.append(signal)
        s1_labels.append(label[0])
        s2_labels.append(label[1])

    signals = np.array(signals)
    s1_labels = np.array(s1_labels,dtype=object)
    s2_labels = np.array(s2_labels,dtype=object)
    filenames = np.array(filenames)

    print("Reading patients done!")

    kf = KFold(CROSS_VALIDATION_FOLD,random_state=0,shuffle=True)

    def calc_patient(i,singal):
        features = pyPCG.lr_hsmm._generate_features(singal.data,SIG_FS,F_FS,preproc=BP)
        print(f"Calculated for {i}")
        return features

    print("Preprocessing patients...")
    tot_features = Parallel(n_jobs=JOBS)(delayed(calc_patient)(i,patients) for i,patients in enumerate(signals))
    tot_features = np.array(tot_features,dtype=object)
    print("Preprocessing done!")

    def cv_fold(i,train_ind,test_ind):
        print(f"Training fold {i}...")
        curr_features = tot_features[train_ind] # type: ignore
        curr_signal = signals[train_ind]
        curr_s1_labels = s1_labels[train_ind]
        curr_s2_labels = s2_labels[train_ind]
        flat_features = [[],[],[],[]]
        for record in curr_features:
            for ind,feature in enumerate(record):
                flat_features[ind] = flat_features[ind]+feature.tolist()

        flat_features = np.array(flat_features)
        curr_signal_data = []
        for signal in curr_signal:
            curr_signal_data.append(signal.data)

        model = pyPCG.lr_hsmm.LR_HSMM()
        model.signal_fs = SIG_FS
        model.feature_fs = F_FS
        model.bandpass_frq = BP
        model.expected_hr_range = HR
        model.mean_s1_len = S1
        model.mean_s2_len = S2
        model.std_s1_len = S1_STD
        model.std_s2_len = S2_STD
        model.train_with_precalc_features(flat_features,curr_signal_data,curr_s1_labels,curr_s2_labels) #type: ignore
        model.save_model(os.path.join(outpath,f"hsmm_trained_{i}.json"))
        print(f"Training {i} done!")
        print(f"Testing fold {i}...")

        s1_pred_out,s2_pred_out,fnames_out = [],[],[]
        test_signals = signals[test_ind]
        test_fnames = filenames[test_ind]
        for t_signal,t_file in zip(test_signals,test_fnames):
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
        print(f"Testing {i} done!")
        return fnames_out,s1_pred_out, s2_pred_out

    print("Starting crossvalidation...")
    results = Parallel(n_jobs=JOBS)(delayed(cv_fold)(i,train,test) for i,(train,test) in enumerate(kf.split(signals)))
    print("Crossvalidation done!")

    for fold in results: #type: ignore
        fnames = fold[0] #type: ignore
        s1_preds = fold[1] #type: ignore
        s2_preds = fold[2] #type: ignore
        for fname,s1_pred,s2_pred in zip(fnames,s1_preds,s2_preds):
            with open(os.path.join(outpath,method,f"{fname[:-4]}.csv"),"w") as out_file:
                out_file.write("Location;Value\n")
                for s in s1_pred:
                    out_file.write(f"{s};S1\n")
                for s in s2_pred:
                    out_file.write(f"{s};S2\n")
    print(f"Ended: {datetime.datetime.now().time()}")
    print("======================================")
