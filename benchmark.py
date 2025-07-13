import os
import pyPCG.io
import calc
import numpy as np
import json
import matplotlib.pyplot as plt
import scipy.stats
import warnings

def rm_outliers(data):
    loc_data = np.array(data)
    limit = scipy.stats.iqr(loc_data)*1.5
    lo = np.percentile(loc_data,25)-limit
    hi = np.percentile(loc_data,75)+limit
    return loc_data[np.logical_and(loc_data>lo,loc_data<hi)]

def get_detections(detection):
    DL_S1, DL_S2 = [], []
    for filename in sorted(os.listdir(detection)):
        if not filename.endswith("csv"):
            continue
        S1, S2 = pyPCG.io.read_hsannot_file(os.path.join(detection,filename))
        DL_S1.append(S1)
        DL_S2.append(S2)
    return DL_S1, DL_S2

def get_tolerance_scores(detect,gt,tols):
    sens,spec,ppv,f1,b_acc = [],[],[],[],[]
    for tol in tols:
        tp,fp,tn,fn = 0,0,0,0
        if isinstance(detect[0], float):
            tp, fp, tn, fn = calc.tolerance_detect(np.array(detect),np.array(gt),tolerance=tol)
        else:
            for GT,DL in zip(gt,detect):
                A = calc.tolerance_detect(np.array(DL),np.array(GT),tolerance=tol)
                tp+=A[0]
                fp+=A[1]
                tn+=A[2]
                fn+=A[3]
        B = calc.acc_measure(tp,fp,tn,fn)
        sens.append(B[0])
        spec.append(B[1])
        ppv.append(B[2])
        f1.append(B[3])
        b_acc.append(B[4])
    return sens,spec,ppv,f1,b_acc

def mae(gt,detect):
    error=[]
    if len(detect)>0 and isinstance(detect[0], float):
        e  = calc.abs_error(np.array(detect)*1000,np.array(gt)*1000)
        error.append(e)
    else:
        for GT,DL in zip(gt,detect):
            e = calc.abs_error(np.array(DL)*1000,np.array(GT)*1000)
            error.append(e)
    m,s,_=calc.mean_std(np.array(error))
    return m,s

def error_rate(gt1,gt2,detect1,detect2,tol=0.03):
    dels,inss,subs = [],[],[]
    if isinstance(detect1[0], float):
        d,i,s = calc.hs_error_rate(np.array(detect1),np.array(detect2),np.array(gt1),np.array(gt2),tol)
        dels.append(d)
        inss.append(i)
        subs.append(s)
    else:
        for GT1,GT2,DL1,DL2 in zip(gt1,gt2,detect1,detect2):
            d,i,s = calc.hs_error_rate(np.array(DL1),np.array(DL2),np.array(GT1),np.array(GT2),tol)
            dels.append(d)
            inss.append(i)
            subs.append(s)
    return dels,inss,subs

def flatten(matrix):
    return [item for row in matrix for item in row]

class method:
    def __init__(self, name, path, S1=None, S2=None) -> None:
        self.name = name
        self.path = path
        
        if S1 is not None:
            self.S1 = S1
        else:
            self.S1 = list()
        if S2 is not None:
            self.S2 = S2
        else:
            self.S2 = list()
        
        self.cache_file = os.path.join(self.path,self.name+".json")
        if os.path.isfile(self.cache_file):
            self.load_cache()
            return

        self.HR = list()
        self.PPV_1 = list()
        self.F1_1 = list()
        self.MAE_1_m = None
        self.MAE_1_std = None
        self.PPV_2 = list()
        self.F1_2 = list()
        self.MAE_2_m = None
        self.MAE_2_std = None
        self.err_dels = list()
        self.err_inss = list()
        self.err_subs = list()
        self.MSE_hr = list()
        self.BA_hr = list()
    
    def load_sounds(self, path):
        self.S1, self.S2 = get_detections(path)
    
    def calc_HR(self, win_len=0, overlap=0.0, bpm_range=(80,210), force=False):
        if self.S1 is None:
            warnings.warn("No S1 information. HR will not be calculated",UserWarning)
            return
        if not force and not len(self.HR)==0:
            return self.HR

        if win_len == 0:
            for s1 in self.S1:
                self.HR.append([len(s1)/s1[-1]*60])
        else:
            if isinstance(self.S1[0], float):
                step = win_len-(win_len*overlap)
                at_end = False
                start = 0
                s = np.array(self.S1)
                hr = list()
                while not at_end:
                    end = start+win_len
                    if end >= s[-1]:
                        end = s[-1]
                        at_end = True
                    subwin = s[s>start]
                    subwin = subwin[subwin<end]
                    s1diff = np.diff(subwin)
                    # Tang range: 80-210 bpm
                    s1diff = s1diff[s1diff<(60/bpm_range[0])]
                    s1diff = s1diff[s1diff>(60/bpm_range[1])]

                    hr.append(60/np.median(s1diff))
                    start += step
                self.HR.append(hr)
            else:
                for s1 in self.S1:
                    step = win_len-(win_len*overlap)
                    at_end = False
                    start = 0
                    s = np.array(s1)
                    hr = list()
                    while not at_end:
                        end = start+win_len
                        if end >= s[-1]:
                            end = s[-1]
                            at_end = True
                        subwin = s[s>start]
                        subwin = subwin[subwin<end]
                        s1diff = np.diff(subwin)
                        # Tang range: 80-210 bpm
                        s1diff = s1diff[s1diff<(60/bpm_range[0])]
                        s1diff = s1diff[s1diff>(60/bpm_range[1])]
                        if len(s1diff) == 0:
                            print("No valid S1-S1 section found. Repeating last HR")
                            hr.append(hr[-1])
                            start += step
                            continue
                        hr.append(60/np.median(s1diff))
                        start += step
                    self.HR.append(hr)
        return self.HR
    
    def calc_tolerance_score(self,GT,tolerance,force=False):
        if not force and not len(self.PPV_1)==0:
            return self.PPV_1, self.F1_1, self.PPV_2, self.F1_2

        _,_,self.PPV_1,self.F1_1,_ = get_tolerance_scores(self.S1,GT[0],tolerance)
        if (len(self.S2)>0 and isinstance(self.S2[0], float)) or len(flatten(self.S2))!=0:
            _,_,self.PPV_2,self.F1_2,_ = get_tolerance_scores(self.S2,GT[1],tolerance)
        return self.PPV_1, self.F1_1, self.PPV_2, self.F1_2
    
    def calc_mae(self,GT,force=False):
        if not force and self.MAE_1_m is not None: #type: ignore
            return self.MAE_1_m,self.MAE_1_std,self.MAE_2_m,self.MAE_2_std
        
        self.MAE_1_m,self.MAE_1_std = mae(GT[0],self.S1)
        if len(flatten(self.S2))!=0:
            self.MAE_2_m,self.MAE_2_std = mae(GT[1],self.S2)
        return self.MAE_1_m,self.MAE_1_std,self.MAE_2_m,self.MAE_2_std
    
    def calc_error(self,GT,tol=0.03,force=False):
        if not force and not len(self.err_dels)==0:
            return self.err_dels,self.err_inss,self.err_subs
        
        self.err_dels,self.err_inss,self.err_subs = error_rate(GT[0],GT[1],self.S1,self.S2,tol)
        return self.err_dels,self.err_inss,self.err_subs
    
    def calc_hr_mse(self,GT,force=False):
        if not force and not len(self.MSE_hr)==0:
            return self.MSE_hr
        
        for hr,gt_hr in zip(self.HR,GT):
            pad_size = len(gt_hr)-len(hr)
            if pad_size>0:
                gt_hr = gt_hr[:-pad_size]
            if pad_size<0:
                hr = hr[:pad_size]
            self.MSE_hr.append(np.mean(np.square(np.array(hr)-np.array(gt_hr))))
        return self.MSE_hr
    
    def cache(self,force=False):
        if not force and os.path.isfile(self.cache_file):
            return
        selfjson = {
            "HR":self.HR,
            "PPV_1":self.PPV_1,
            "F1_1":self.F1_1,
            "MAE_1_m":self.MAE_1_m,
            "MAE_1_std":self.MAE_1_std,
            "PPV_2":self.PPV_2,
            "F1_2":self.F1_2,
            "MAE_2_m":self.MAE_2_m,
            "MAE_2_std":self.MAE_2_std,
            "err_dels":self.err_dels,
            "err_inss":self.err_inss,
            "err_subs":self.err_subs,
            "MSE_hr":self.MSE_hr,
        }
        with open(self.cache_file,"w") as cachefile:
            cachefile.write(json.dumps(selfjson))
    
    def load_cache(self):
        with open(self.cache_file,"r") as cachefile:
            selfjson = json.loads(cachefile.read())
            self.HR = selfjson["HR"]
            self.PPV_1 = selfjson["PPV_1"]
            self.F1_1 = selfjson["F1_1"]
            self.MAE_1_m = selfjson["MAE_1_m"]
            self.MAE_1_std = selfjson["MAE_1_std"]
            self.PPV_2 = selfjson["PPV_2"]
            self.F1_2 = selfjson["F1_2"]
            self.MAE_2_m = selfjson["MAE_2_m"]
            self.MAE_2_std = selfjson["MAE_2_std"]
            self.err_dels = selfjson["err_dels"]
            self.err_inss = selfjson["err_inss"]
            self.err_subs = selfjson["err_subs"]
            self.MSE_hr = selfjson["MSE_hr"]
    
    def print_report(self,tolerances,tol=0.03):
        ind = np.nonzero(tolerances>tol)[0][0]
        
        print(f"--- {self.name}")
        if (len(self.S1)>0 and isinstance(self.S1[0],float)) or len(flatten(self.S1))!=0:
            print("     PPV\tF1\tMAE")
            print(f" S1: {self.PPV_1[ind]:.3f}\t{self.F1_1[ind]:.3f}\t{self.MAE_1_m:.1f}±{self.MAE_1_std:.1f}")
            if (len(self.S2)>0 and isinstance(self.S2[0], float)) or len(flatten(self.S2))!=0:
                print(f" S2: {self.PPV_2[ind]:.3f}\t{self.F1_2[ind]:.3f}\t{self.MAE_2_m:.1f}±{self.MAE_2_std:.1f}")
            print()
        if len(self.err_dels) != 0:
            print("     Del\tIns\tSub")
            print(f"ERR: {np.sum(self.err_dels)}\t{np.sum(self.err_inss)}\t{np.sum(self.err_subs)}")
            print()
        if len(self.MSE_hr) != 0:
            print(f"HR MSE: {np.mean(self.MSE_hr):.3f}±{np.std(self.MSE_hr):.3f} [{np.min(self.MSE_hr):.3f}-{np.max(self.MSE_hr):.3f}] IQR: {np.subtract(*np.percentile(self.MSE_hr, [75, 25])):.3f}") #{scipy.stats.trim_mean(self.MSE_hr,0.05):.3f}
        print()

def plot_methods(methods,tolerances,score="F1_1"):
    scores = [m.F1_1 for m in methods]
    names = [m.name for m in methods]
    plt.figure(figsize=(7,4.1))
    lines = custom_lines()
    for s,n in zip(scores,names):
        style, color = next(lines)
        plt.plot(tolerances*1000,s,linestyle=style,color=color,linewidth=2)
    plt.ylim((0,1.02))
    ax = plt.gca()
    ax.set_xticks(np.arange(0, tolerances[-1]*1000, 20))
    ax.set_xticks(np.arange(0, tolerances[-1]*1000, 10), minor = True)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_yticks(np.arange(0, 1.1, 0.1), minor = True)
    plt.grid(True,"both")
    ax.grid(which="minor",alpha=0.3,linestyle="--")
    plt.xlim((tolerances[0]*1000-3,tolerances[-1]*1000+2))
    plt.xlabel("Tolerance [ms]",fontsize="large")
    plt.ylabel(score,fontsize="large")
    plt.legend(names,loc=(0.65,0.05),fontsize="medium")
    plt.show()

def plot_BA(method,GT):
    means, diffs = list(),list()
    for hr, gt_hr in zip(method.HR,GT):
        pad_size = len(gt_hr)-len(hr)
        if pad_size>0:
            gt_hr = gt_hr[:-pad_size]
        if pad_size<0:
            hr = hr[:pad_size]
        hr, gt_hr = np.array(hr,dtype=float), np.array(gt_hr,dtype=float)
        d = gt_hr-hr
        m = (gt_hr+hr)/2
        means += m.tolist()
        diffs += d.tolist()
        
    means, diffs = np.array(means), np.array(diffs)
    mdiff = np.mean(diffs)
    sdiff = np.std(diffs)
    
    xlim = (110,170)
    ylim = (-20,20)
    plt.figure(figsize=(7,5))
    plt.scatter(means,diffs,facecolors='none', edgecolors='r')
    plt.axhline(mdiff,color="b") #type: ignore
    # plt.axhline(trim_md,color="g") #type: ignore
    plt.axhline((mdiff+sdiff*1.96),linestyle="--",color="b") #type: ignore
    # plt.axhline((trim_md+trim_sd*1.96),linestyle="--",color="g") #type: ignore
    plt.text(147,6.5,f"{(mdiff+sdiff*1.96):.2f}",fontsize=14,color="b")
    plt.axhline((mdiff-sdiff*1.96),linestyle="--",color="b") #type: ignore
    # plt.axhline((trim_md-trim_sd*1.96),linestyle="--",color="g") #type: ignore
    plt.text(147,-5.5,f"{(mdiff-sdiff*1.96):.2f}",fontsize=14,color="b")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.gca().tick_params(labelsize="large")
    plt.ylabel("Difference of values [BPM]",fontsize="xx-large")
    plt.xlabel("Mean of values [BPM]",fontsize="xx-large")
    plt.title(method.name,fontsize=24)
    plt.tight_layout()
    
def custom_lines():
    color_cycle = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
    styles = ["-","--","-.",":"]
    for style in styles:
        for color in color_cycle:
            yield (style, color)
