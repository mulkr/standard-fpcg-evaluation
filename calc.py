import numpy as np
import numpy.typing as npt

def mean_std(dat:npt.NDArray) -> tuple[float,float,str]:
    m,s = np.mean(dat), np.std(dat)
    fstr = f"{m:.1f}±{s:.1f}"
    return m,s,fstr # type: ignore

def tolerance_detect(detect: npt.NDArray[np.int_|np.float_], label: npt.NDArray[np.int_|np.float_], tolerance: float=0.06) -> tuple[int,int,int,int]:
    difs = []
    for det in detect:
        temp = np.abs(label-det)
        difs.append(np.min(temp))
    difs = np.array(difs)
    tp = np.sum(difs<tolerance).astype(int)
    fp = np.sum(difs>=tolerance).astype(int)
    difs = []
    for lab in label:
        temp = np.abs(detect-lab)
        # if np.min(temp)>1000:
        #     continue
        difs.append(np.min(temp))
    difs = np.array(difs)
    fn = np.sum(difs>=tolerance).astype(int)
    # tp = np.sum(np.min(np.abs(np.subtract.outer(label,detect)),axis=0)<tolerance)
    # fp = np.sum(np.min(np.abs(np.subtract.outer(label,detect)),axis=0)>=tolerance)
    # fn = len(label)-tp
    tn = 0
    return tp,fp,tn,fn

def acc_measure(tp:int=0,fp:int=0,tn:int=0,fn:int=0) -> tuple[float,float,float,float,float]:
    sens = tp/(tp+fn)
    spec = tn/(tn+fp)
    ppv = tp/(tp+fp)
    f1 = (2*sens*ppv)/(sens+ppv)
    b_acc = (sens+spec)/2
    return sens,spec,ppv,f1,b_acc

def abs_error(detect: npt.NDArray[np.int_|np.float_], label: npt.NDArray[np.int_|np.float_]) -> tuple[float,float]:
    difs = []
    for det in detect:
        temp = np.abs(label-det)
        difs.append(np.min(temp))
        
    mae = np.mean(difs).astype(float)
    return mae

def rel_error(detect: npt.NDArray[np.int_|np.float_], label: npt.NDArray[np.int_|np.float_]) -> npt.NDArray[np.float_]:
    dist = np.diff(label)
    dist = np.append(dist,dist[-1])
    diff = np.min(np.abs(np.subtract.outer(label,detect)),axis=0)
    ind = np.argmin(np.abs(np.subtract.outer(label,detect)),axis=0)
    return diff/dist[ind]

def hs_error_rate(detect_s1, detect_s2, label_s1, label_s2, tolerance):
    sub = 0
    ins = 0
    dlt = 0
    for i in range(len(detect_s1)-1):
        d = detect_s1[i]
        d_n = detect_s1[i+1]
        start = d-tolerance
        end = d+tolerance
        start_n = d_n-tolerance
        s1_in = np.logical_and(label_s1<end,label_s1>start)
        s2_in = np.logical_and(label_s2<end,label_s2>start)
        s1_insert = np.logical_and(label_s1>end,label_s1<start_n)
        if not (np.any(s1_in) or np.any(s2_in)):
            dlt += 1
        if np.any(s1_insert):
            ins += np.count_nonzero(s1_insert)
        if np.any(s2_in):
            sub += np.count_nonzero(s2_in)

    for i in range(len(detect_s2)-1):
        d = detect_s2[i]
        d_n = detect_s2[i+1]
        start = d-tolerance
        end = d+tolerance
        start_n = d_n-tolerance
        s1_in = np.logical_and(label_s1<end,label_s1>start)
        s2_in = np.logical_and(label_s2<end,label_s2>start)
        s2_insert = np.logical_and(label_s2>end,label_s2<start_n)
        if not (np.any(s1_in) or np.any(s2_in)):
            dlt += 1
        if np.any(s2_insert):
            ins += np.count_nonzero(s2_insert)
        if np.any(s1_in):
            sub += np.count_nonzero(s1_in)
    
    return dlt, ins, sub