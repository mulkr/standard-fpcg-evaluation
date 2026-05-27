import benchmark
import numpy as np
import json
from joblib import Parallel, delayed
import os

TOL = 0.03 # tolerance value used for accuracy metrics
HR_WIN = 10 # time window in secods to calculate heart rate
HR_WIN_OVERLAP = 0.5 # heart rate window overlap percentage
HR_RANGE = (100,210) # range of acceptable heart rates
JOBS = 4 # evaluation jobs to run in parallel

noise_gt_path = "/home/mulkr/PhD/fpcg-gen/noise_levels"
noise_results_path = "/home/mulkr/PhD/fpcg-gen/noise_testing"

def do_benchmark(result_path,gt_path,noise_level):
    # Ground truth
    GT = benchmark.get_detections(gt_path)
    gt_m = benchmark.method("GroundTruth",*GT)
    gt_m.calc_HR(HR_WIN,HR_WIN_OVERLAP,HR_RANGE)

    # Load methods for testing
    # Make sure the detections are in the correct folder and use the pyPCG format
    current_methods = [
        benchmark.method("Müller",*benchmark.get_detections(os.path.join(result_path,"muller_pretrained"))),
        benchmark.method("Springer",*benchmark.get_detections(os.path.join(result_path,"springer"))),
        benchmark.method("Cesarelli",*benchmark.get_detections(os.path.join(result_path,"cesarelli"))),
        benchmark.method("Balogh",*benchmark.get_detections(os.path.join(result_path,"balogh"))),
        benchmark.method("Schmidt",*benchmark.get_detections(os.path.join(result_path,"schmidt"))),
        benchmark.method("Renna-Seqmax",*benchmark.get_detections(os.path.join(result_path,"renna_seqmax_pretrained"))),
        benchmark.method("Renna-HSMM",*benchmark.get_detections(os.path.join(result_path,"renna_hsmm_pretrained"))),
        benchmark.method("Chen",*benchmark.get_detections(os.path.join(result_path,"chen"))),
    ]

    tolerances = np.arange(0.003,0.09,0.003)

    # Load Tang et al. FHR manually
    tang = benchmark.method("Tang")
    with open(os.path.join(result_path,"tang_fhr.json"),"r") as tang_hr:
        hrs = json.loads(tang_hr.read())
        for hr in hrs.values():
            tang.HR.append(hr)

    # Load Zahorian et al. FHR manually
    zahorian = benchmark.method("Zahorian")
    with open(os.path.join(result_path,"zahorian_fhr.json"),"r") as zahorian_hr:
        hrs = json.loads(zahorian_hr.read())
        for hr in hrs.values():
            zahorian.HR.append(hr)

    # Calculation function
    def calc_measures(method: benchmark.method):
        method.calc_HR(HR_WIN,HR_WIN_OVERLAP,HR_RANGE)
        method.calc_mae(GT)
        method.calc_tolerance_score(GT,tolerances)
        method.calc_error(GT,TOL)
        method.calc_hr_mse(gt_m.HR)
        return method

    # Calculate measures in paralell
    # with tqdm_joblib(total=len(current_methods)):
    methods: list[benchmark.method] = Parallel(n_jobs=JOBS)(delayed(calc_measures)(m) for m in current_methods) #type:ignore
    tang.calc_hr_mse(gt_m.HR)
    zahorian.calc_hr_mse(gt_m.HR)

    methods += [tang,zahorian]

    # Print measures
    for c_method in methods:
        c_method.file_report(f"{noise_level}.txt",tolerances,TOL)

for noise_level, gt_path in zip(sorted(os.listdir(noise_results_path)),sorted(os.listdir(noise_gt_path))):
    print(os.path.join(noise_gt_path,gt_path),noise_level)
    if os.path.isfile(f"{noise_level}.txt"):
        os.remove(f"{noise_level}.txt")
    do_benchmark(os.path.join(noise_results_path,noise_level),os.path.join(noise_gt_path,gt_path),noise_level)
