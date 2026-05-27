import os
import pyPCG.io
import calc
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import scipy.stats
import warnings

def rm_outliers(data: npt.NDArray[np.float_|np.int_]|list[float|int]) -> npt.NDArray[np.float_|np.int_]:
    """Remove outliers from data based on interquartile range (outlier<first quartile-iqr*1.5 or outlier>third quartile-iqr*1.5)

    Args:
        data (np.ndarray): input data to remove outliers from

    Returns:
        np.ndarray: data with outliers removed
    """
    loc_data = np.array(data)
    limit = scipy.stats.iqr(loc_data)*1.5
    lo = np.percentile(loc_data,25)-limit
    hi = np.percentile(loc_data,75)+limit
    return loc_data[np.logical_and(loc_data>lo,loc_data<hi)]

def get_detections(detection: str) -> tuple[list[list[float]],list[list[float]]]:
    """Load detection results from multiple files

    Args:
        detection (str): directory containing the detections

    Returns:
        list[list[float]]: detections for S1
        list[list[float]]: detections for S2
    """
    DL_S1, DL_S2 = [], []
    for filename in sorted(os.listdir(detection)):
        if not filename.endswith("csv"):
            continue
        S1, S2 = pyPCG.io.read_hsannot_file(os.path.join(detection,filename))
        DL_S1.append(S1)
        DL_S2.append(S2)
    return DL_S1, DL_S2

def get_tolerance_scores(detect,gt,tols) -> tuple[list[float],list[float],list[float],list[float],list[float]]:
    """Calculate accuracy scores iterated through different tolerance windows

    Args:
        detect: detections
        gt: ground truth
        tols: tolerance window sizes

    Returns:
        list[float]: sensitivities
        list[float]: specificities
        list[float]: positive predictive values
        list[float]: f1-scores
        list[float]: balanced accuracies
    """
    sens,spec,ppv,f1,b_acc = [],[],[],[],[]
    for tol in tols:
        tp,fp,tn,fn = 0,0,0,0
        # if isinstance(detect[0], float): # is this necessary?
            # tp, fp, tn, fn = calc.tolerance_detect(np.array(detect),np.array(gt),tolerance=tol)
        # else:
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

def mae(gt,detect) -> tuple[float,float]:
    """Calculate all mean absolute errors between ground truth and detections

    Args:
        gt: ground truth
        detect: detections

    Returns:
        float: mean of MAE in ms
        float: standard deviation of MAE ms
    """
    error=[]
    # if len(detect)>0 and isinstance(detect[0], float): # is this necessary?
        # e  = calc.abs_error(np.array(detect)*1000,np.array(gt)*1000)
        # error.append(e)
    # else:
    for GT,DL in zip(gt,detect):
        e = calc.abs_error(np.array(DL)*1000,np.array(GT)*1000)
        error.append(e)
    m,s,_=calc.mean_std(np.array(error))
    return m,s

def error_rate(gt1,gt2,detect1,detect2,tol=0.03) -> tuple[list[int],list[int],list[int]]:
    """Calculate all error rates similar to 'word error rate'

    Args:
        gt1: ground truth S1 locations in seconds
        gt2: ground truth S2 locations in seconds
        detect1: detections S1 locations in seconds
        detect2: detections S2 locations in seconds
        tol: tolerance window size in seconds

    Returns:
        float: mean of MAE in ms
        float: standard deviation of MAE ms
    """
    dels,inss,subs = [],[],[]
    # if isinstance(detect1[0], float): # is this necessary?
    #     d,i,s = calc.hs_error_rate(np.array(detect1),np.array(detect2),np.array(gt1),np.array(gt2),tol)
    #     dels.append(d)
    #     inss.append(i)
    #     subs.append(s)
    # else:
    for GT1,GT2,DL1,DL2 in zip(gt1,gt2,detect1,detect2):
        d,i,s = calc.hs_error_rate(np.array(DL1),np.array(DL2),np.array(GT1),np.array(GT2),tol)
        dels.append(d)
        inss.append(i)
        subs.append(s)
    return dels,inss,subs

def flatten(matrix):
    """Utility function to flatten a list of lists

    Args:
        matrix (list[list]): input to be flattened

    Returns:
        list: flattened matrix
    """
    return [item for row in matrix for item in row]

class method:
    """
    Object representing a detection method and its accuracy scores

    Attributes:
        S1: S1 locations for each record in seconds
        S2: S2 locations for each record in seconds
        HR: heart rate calculated from S1 locations
        PPV_1: positive predictive values for S1
        F1_1: F1-scores for S1
        MAE_1_m: mean of MAE for S1
        MAE_1_std: standard deviation of MAE for S1
        PPV_2: positive predictive values for S2
        F1_2: F1-scores for S2
        MAE_2_m: mean of MAE for S1
        MAE_2_std: standard deviation of MAE for S2
        err_dels: deletion errors for each record
        err_inss: insertion errors for each record
        err_subs: substitusion errors for each record
        MSE_hr: mean square error of HR for each record
        (possibly more)
    """
    def __init__(self, name: str, S1=None, S2=None) -> None:
        self.name = name

        if S1 is not None:
            self.S1 = S1
        else:
            self.S1 = list()
        if S2 is not None:
            self.S2 = S2
        else:
            self.S2 = list()

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
        self.MSE_hr: list[float] = list()

    def load_sounds(self, path):
        self.S1, self.S2 = get_detections(path)

    def calc_HR(self, win_len=0, overlap=0.0, bpm_range=(80,210)):
        # Tang range: 80-210 bpm
        if self.S1 is None:
            warnings.warn("No S1 information. HR will not be calculated",UserWarning)
            return

        if win_len == 0:
            for s1 in self.S1:
                self.HR.append([len(s1)/s1[-1]*60])
        else:
            # if isinstance(self.S1[0], float): # is this necessary? # S1 1D array -> When?
            #     step = win_len-(win_len*overlap)
            #     at_end = False
            #     start = 0
            #     s = np.array(self.S1)
            #     hr = list()
            #     while not at_end:
            #         end = start+win_len
            #         if end >= s[-1]:
            #             end = s[-1]
            #             at_end = True
            #         subwin = s[s>start]
            #         subwin = subwin[subwin<end]
            #         s1diff = np.diff(subwin)
            #         s1diff = s1diff[s1diff<(60/bpm_range[0])]
            #         s1diff = s1diff[s1diff>(60/bpm_range[1])]

            #         hr.append(60/np.median(s1diff))
            #         start += step
            #     self.HR.append(hr)
            # else:
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
                    s1diff = s1diff[s1diff<(60/bpm_range[0])]
                    s1diff = s1diff[s1diff>(60/bpm_range[1])]
                    if len(s1diff) == 0:
                        # print(f"{self.name}: No valid S1-S1 section found. Repeating last HR")
                        if len(hr)>0:
                            hr.append(hr[-1])
                        else:
                            pass
                            # print("No HR to repeat!")
                        start += step
                        continue
                    hr.append(60/np.median(s1diff))
                    start += step
                self.HR.append(hr)
        return self.HR

    def calc_tolerance_score(self,GT,tolerance):
        _,_,self.PPV_1,self.F1_1,_ = get_tolerance_scores(self.S1,GT[0],tolerance)
        if (len(self.S2)>0 and isinstance(self.S2[0], float)) or len(flatten(self.S2))!=0:
            _,_,self.PPV_2,self.F1_2,_ = get_tolerance_scores(self.S2,GT[1],tolerance)
        return self.PPV_1, self.F1_1, self.PPV_2, self.F1_2

    def calc_mae(self,GT):
        self.MAE_1_m,self.MAE_1_std = mae(GT[0],self.S1)
        if len(flatten(self.S2))!=0:
            self.MAE_2_m,self.MAE_2_std = mae(GT[1],self.S2)
        return self.MAE_1_m,self.MAE_1_std,self.MAE_2_m,self.MAE_2_std

    def calc_error(self,GT,tol=0.03):
        self.err_dels,self.err_inss,self.err_subs = error_rate(GT[0],GT[1],self.S1,self.S2,tol)
        return self.err_dels,self.err_inss,self.err_subs

    def calc_hr_mse(self,GT):
        for hr,gt_hr in zip(self.HR,GT):
            pad_size = len(gt_hr)-len(hr)
            if pad_size>0:
                gt_hr = gt_hr[:-pad_size]
            if pad_size<0:
                hr = hr[:pad_size]
            self.MSE_hr.append(np.mean(np.abs(np.array(hr)-np.array(gt_hr)))) #type:ignore
        return self.MSE_hr

    def file_report(self,path,tolerances,tol=0.03):
        ind = np.nonzero(tolerances>tol)[0][0]
        with open(path,"a") as reportfile:
            reportfile.write(f"--- {self.name}\n")
            if (len(self.S1)>0 and isinstance(self.S1[0],float)) or len(flatten(self.S1))!=0:
                reportfile.write("     PPV\tF1\tMAE\n")
                reportfile.write(f" S1: {self.PPV_1[ind]:.3f}\t{self.F1_1[ind]:.3f}\t{self.MAE_1_m:.1f}±{self.MAE_1_std:.1f}\n")
                if (len(self.S2)>0 and isinstance(self.S2[0], float)) or len(flatten(self.S2))!=0:
                    reportfile.write(f" S2: {self.PPV_2[ind]:.3f}\t{self.F1_2[ind]:.3f}\t{self.MAE_2_m:.1f}±{self.MAE_2_std:.1f}\n")
                reportfile.write("\n")
            if len(self.err_dels) != 0:
                reportfile.write("     Del\tIns\tSub\n")
                reportfile.write(f"ERR: {np.sum(self.err_dels)}\t{np.sum(self.err_inss)}\t{np.sum(self.err_subs)}\n")
                reportfile.write("\n\n")
            if len(self.MSE_hr) != 0:
                reportfile.write(f"HR MSE: {np.mean(self.MSE_hr):.3f}±{np.std(self.MSE_hr):.3f} [{np.min(self.MSE_hr):.3f}-{np.max(self.MSE_hr):.3f}] IQR: {np.subtract(*np.percentile(self.MSE_hr, [75, 25])):.3f}\n")
            reportfile.write("\n")

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
