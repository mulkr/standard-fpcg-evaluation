import os
import sys
import datetime
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
import scipy.io.wavfile
from preprocess import renna_preprocess_wave, check_valid_sequence
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore
import tensorflow as tf
from hsmmlearn.emissions import AbstractEmissions
from hsmmlearn.hsmm import HSMMModel
from scipy.stats import multivariate_normal
from pyPCG.lr_hsmm import _generate_states, _get_hr_sys, _get_duration_distributions
from pyPCG.segment import heart_state, convert_hsmm_states
from pyPCG.io import read_hsannot_file

CROSS_VALIDATION_FOLD = 10
JOBS = 5
model_name = "rbio3.9_BP_15-100"

SIG_FS = 1000
F_FS = 50
BP = (15,55)
S1_L = 70
S2_L = 65
HR = (100,200)
S1_STD = 22
S2_STD = 22

# parameters of the neural network and timing
N = 64
tau = N//8
lr = 1e-6

def align_timing(start,end):
    a_start, a_end = start,end
    if start[0] > end[0]:
        a_end = end[1:]
    if start[-1] > end[-1]:
        a_start = start[:-1]
    return a_start, a_end


class CNNEmission(AbstractEmissions):
    def __init__(self,cnn_model):
        self.cnn = cnn_model

        # precalculated from training data
        self.mu = np.array([5.71449701e-16, 1.47697907e-15, 2.17594620e-16, -4.89082078e-16])
        self.sigma = np.array([[ 1.00001356, 0.9378088, 0.80308249, -0.09567174],
                               [ 0.9378088, 1.00001356, 0.87772103, -0.05771865],
                               [ 0.80308249, 0.87772103, 1.00001356, -0.02121787],
                               [-0.09567174, -0.05771865, -0.02121787, 1.00001356]])

    def likelihood(self, obs):
        tobs = obs.T
        work = list()
        win_size = 64
        stride = win_size//8
        for win in range(0,tobs.shape[1]-win_size,stride):
            temp = tobs[:,win:(win+win_size)]
            if temp.shape[1]<win_size:
                break
            work.append(temp.T)
        work = np.array(work)

        pred = self.cnn.predict(work) # type: ignore

        pred_shape = np.reshape(pred,(-1,4))
        n_windows = pred_shape.shape[0]//win_size
        s_len = stride*(n_windows-1)+win_size
        s_expand = np.full((n_windows,s_len,4),np.nan)
        for i in range(n_windows):
            s_expand[i, stride*i:stride*i+win_size, :] = pred_shape[(i*win_size):(i*win_size)+win_size, :]
        s = np.nanmean(s_expand, axis=0)

        prob = np.squeeze(s)

        probs = np.zeros_like(prob)
        for n in range(4):
            pi_hat = prob[:,n]
            for t in range(len(pi_hat)):
                correction = multivariate_normal.pdf(obs[t,:],mean=self.mu,cov=self.sigma) #type:ignore
                probs[t,n] = (pi_hat[t]*correction)/0.25

        pad_size = max(obs.shape[0]-probs.shape[0],0)
        pad = np.zeros((pad_size,4))
        probs = np.concatenate((probs,pad),axis=0)

        return probs.T.astype(np.double)

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
    print(f"Cross validation CNN-Seqmax - Folds: {CROSS_VALIDATION_FOLD} - Processing jobs: {JOBS}")
    print(f"Started: {datetime.datetime.now().time()}")

    print("Reading patient data...")

    signals = []
    s1_labels,s2_labels = [],[]
    filenames = []

    for filename in os.listdir(datapath):
        if not filename.endswith("wav"):
            continue
        fs, data = scipy.io.wavfile.read(os.path.join(datapath,filename))
        label = read_hsannot_file(os.path.join(labelpath,f"{filename[:-4]}.csv"))
        print(filename)
        filenames.append(filename)
        signal = data - np.mean(data)
        signal = signal / np.max(np.abs(signal))
        signals.append(signal)
        s1_labels.append(label[0])
        s2_labels.append(label[1])

    signals = np.array(signals,dtype=object)
    s1_labels = np.array(s1_labels,dtype=object)
    s2_labels = np.array(s2_labels,dtype=object)
    filenames = np.array(filenames)

    print("Reading patients done!")

    kf = KFold(CROSS_VALIDATION_FOLD,random_state=0,shuffle=True)

    def cv_fold(i,train_ind,test_ind):
        print(f"Training fold {i}...")

        model= tf.keras.models.load_model(f"{model_name}.h5") # type:ignore

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), #type:ignore
                    loss=tf.keras.losses.CategoricalCrossentropy(), # type:ignore
                    metrics=[tf.keras.metrics.CategoricalAccuracy()]) # type:ignore

        curr_signal = signals[train_ind]
        curr_s1_labels = s1_labels[train_ind]
        curr_s2_labels = s2_labels[train_ind]

        train_data = list()
        train_label = list()
        for sig,s1,s2 in zip(curr_signal,curr_s1_labels,curr_s2_labels):
            states = _generate_states(sig,s1,s2,SIG_FS,F_FS)

            env = renna_preprocess_wave(sig,SIG_FS,F_FS)

            for win in range(0,env.shape[1]-N,tau):
                temp = states[win:(win+N)]-1
                if len(temp)<64:
                    break
                train_data.append(env[:,win:(win+N)].T)
                train_label.append(temp)

        train_data = np.array(train_data)
        train_label = np.array(train_label)+1

        train_data, train_label = check_valid_sequence(train_data,train_label,verbose=0)
        train_label = to_categorical(train_label-1)

        callbacks = []
        checkpoint = ModelCheckpoint(f'{model_name}_{i}_tuned_hsmm.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks.append(checkpoint)

        history_finetune = model.fit(train_data,train_label,epochs=15,callbacks=callbacks,verbose=2) #type:ignore

        print(f"Training {i} done!")
        print(f"Testing fold {i}...")

        s1_pred_out,s2_pred_out,fnames_out = [],[],[]
        test_signals = signals[test_ind]
        test_fnames = filenames[test_ind]
        test_model = tf.keras.models.load_model(f"{model_name}_{i}_tuned_hsmm.h5") #type:ignore
        tmat = np.array([[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.],[1.,0.,0.,0.]])
        emit = CNNEmission(test_model)
        for t_signal,t_file in zip(test_signals,test_fnames):
            env = renna_preprocess_wave(t_signal,SIG_FS,F_FS)

            hr, sys = _get_hr_sys(t_signal,SIG_FS,BP,HR[0],HR[1])
            durs = _get_duration_distributions(hr,sys,F_FS,S1_L,S2_L,S1_STD,S2_STD)
            hsmm_model = HSMMModel(emit,durs,tmat)

            state = hsmm_model.decode(env.T)

            pred_s1s,pred_s1e = convert_hsmm_states(state+1,state_id=heart_state.S1)
            pred_s2s,pred_s2e = convert_hsmm_states(state+1,state_id=heart_state.S2)

            pred_s1s, pred_s1e = align_timing(pred_s1s,pred_s1e)
            pred_s2s, pred_s2e = align_timing(pred_s2s,pred_s2e)

            pred_s1 = (pred_s1s+pred_s1e)/2
            pred_s2 = (pred_s2s+pred_s2e)/2

            s1_pred_out.append(pred_s1/F_FS)
            s2_pred_out.append(pred_s2/F_FS)
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

