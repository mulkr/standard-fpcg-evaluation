import numpy as np
import numpy.typing as npt

def mean_std(dat:npt.NDArray) -> tuple[float,float,str]:
    """Calculate mean and standard deviation and format a string with these data

    Args:
        dat (np.ndarray): input data to calculate mean and standard deviation

    Returns:
        float: calculated mean
        float: calculated standard deviation
        str: the formatted string with one digit after the decimal point (<mean>±<std>)
    """
    m,s = np.mean(dat), np.std(dat)
    fstr = f"{m:.1f}±{s:.1f}"
    return m,s,fstr # type: ignore

def tolerance_detect(detect: npt.NDArray[np.int_|np.float_], label: npt.NDArray[np.int_|np.float_], tolerance: float=0.06) -> tuple[int,int,int,int]:
    """Convert detections to true positive, false positive, true negative, false negative classes based on a tolerance value

    Args:
        detect (np.ndarray): detection locations in seconds
        label (np.ndarray): ground truth label locations in seconds
        tolerance (float): tolerance window size in seconds

    Returns:
        int: number of true positives
        int: number of false positives
        int: number of true negatives
        int: number of false negatives
    """
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
    """Calculate different accuracy measures from true positive, etc. classes

    Args:
        tp (int): number of true positives
        fp (int): number of false positives
        tn (int): number of true negatives
        fn (int): number of false negatives

    Returns:
        float: sensitivity
        float: specificity
        float: positive predictive value
        float: f1-score
        float: balanced accuracy
    """
    sens = tp/(tp+fn)
    spec = tn/(tn+fp)
    ppv = tp/(tp+fp)
    f1 = (2*sens*ppv)/(sens+ppv)
    b_acc = (sens+spec)/2
    return sens,spec,ppv,f1,b_acc

def abs_error(detect: npt.NDArray[np.int_|np.float_], label: npt.NDArray[np.int_|np.float_]) -> float:
    """Calculate mean absolute error between detected locations and ground truth

    Args:
        detect (np.ndarray): detection locations in seconds
        label (np.ndarray): ground truth label locations in seconds

    Returns:
        float: mean absolute error between detections and ground truth
    """
    difs = []
    for det in detect:
        temp = np.abs(label-det)
        difs.append(np.min(temp))

    mae = np.mean(difs).astype(float)
    return mae

# def rel_error(detect: npt.NDArray[np.int_|np.float_], label: npt.NDArray[np.int_|np.float_]) -> npt.NDArray[np.float_]:
#     dist = np.diff(label)
#     dist = np.append(dist,dist[-1])
#     diff = np.min(np.abs(np.subtract.outer(label,detect)),axis=0)
#     ind = np.argmin(np.abs(np.subtract.outer(label,detect)),axis=0)
    # return diff/dist[ind]

def hs_error_rate(detect_s1: npt.NDArray[np.int_|np.float_], detect_s2: npt.NDArray[np.int_|np.float_], label_s1: npt.NDArray[np.int_|np.float_], label_s2: npt.NDArray[np.int_|np.float_], tolerance: float) -> tuple[int,int,int]:
    """Calculate calculate error rates similar to a 'word error rate'

    Args:
        detect_s1 (np.ndarray): S1 detection locations in seconds
        detect_s2 (np.ndarray): S2 detection locations in seconds
        label_s1 (np.ndarray): S1 ground truth label locations in seconds
        label_s2 (np.ndarray): S2 ground truth label locations in seconds
        tolerance (float): tolerance window size in seconds

    Returns:
        int: number of deletions
        int: number of insertions
        int: number of substitutions
    """
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
