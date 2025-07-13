import benchmark
import numpy as np
import json
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
import matplotlib.pyplot as plt

TOL = 0.03 # tolerance value used for accuracy metrics
HR_WIN = 10 # time window in secods to calculate heart rate
HR_WIN_OVERLAP = 0.5 # heart rate window overlap percentage
JOBS = 4 # evaluation jobs to run in parallel

# Ground truth
GT = benchmark.get_detections("C:/Users/mulkr/PhD/top50/top50label")
gt_m = benchmark.method("GroundTruth","",*GT)
gt_m.calc_HR(HR_WIN,HR_WIN_OVERLAP)


# Load methods for testing
# Make sure the detections are in the correct folder and use the pyPCG format
current_methods = [
    benchmark.method("Müller","./cache",*benchmark.get_detections("./CV_HSMM")),
    benchmark.method("Springer","./cache",*benchmark.get_detections("./cv_springer")),
    benchmark.method("Cesarelli","./cache",*benchmark.get_detections("./cesarelli")),
    benchmark.method("Balogh","./cache",*benchmark.get_detections("./balogh")),
    benchmark.method("Schmidt","./cache",*benchmark.get_detections("./schmidt")),
    benchmark.method("Renna-1","./cache",*benchmark.get_detections("./CNN/finetune/Seqmax")),
    benchmark.method("Renna-2","./cache",*benchmark.get_detections("./CNN/finetune/HSMM")),
    benchmark.method("Chen","./cache",*benchmark.get_detections("./chen")),
]

tolerances = np.arange(0.003,0.09,0.003)

# Load Tang et al. FHR manually
tang = benchmark.method("Tang","./cache")
with open("tang.json","r") as tang_hr:
    hrs = json.loads(tang_hr.read())
    for hr in hrs.values():
        tang.HR.append(hr)

# Load Zahorian et al. FHR manually
zahorian = benchmark.method("Zahorian","./cache")
with open("zahorian.json","r") as zahorian_hr:
    hrs = json.loads(zahorian_hr.read())
    for hr in hrs.values():
        zahorian.HR.append(hr)

# Calculation function
# Uncomment the cache line to speed up re-evaluation
# There could be some problems using the cache. Make sure that recalculation was done if something seems wrong
def calc_measures(method):
    method.calc_HR(HR_WIN,HR_WIN_OVERLAP)
    method.calc_mae(GT)
    method.calc_tolerance_score(GT,tolerances)
    method.calc_error(GT,TOL)
    method.calc_hr_mse(gt_m.HR)
    # method.cache()
    return method

# Calculate measures in paralell
with tqdm_joblib(total=len(current_methods)):
    methods: list[benchmark.method] = Parallel(n_jobs=JOBS)(delayed(calc_measures)(m) for m in current_methods) # type: ignore
tang.calc_hr_mse(gt_m.HR)
zahorian.calc_hr_mse(gt_m.HR)

# Plot F1 Score-vs-Tolerance
benchmark.plot_methods(methods,tolerances)

methods += [tang,zahorian]

# Print measures
for c_method in methods:
    c_method.print_report(tolerances,TOL)

# Settings for plotting
max_error = 10
names = [m.name for m in methods]
fhr_errors = [m.MSE_hr for m in methods]

# Calculate outliers outside plot area
for method in methods:
    temp = np.array(method.MSE_hr)
    print(method.name,"outliers:",np.count_nonzero(temp>max_error))

# Create violin plot similar to the one in the article
plt.boxplot(fhr_errors,showcaps=False,showbox=False,showmeans=False)
plt.violinplot([benchmark.rm_outliers(error) for error in fhr_errors],showmeans=True)
plt.xticks(ticks=(np.arange(len(names))+1),labels=names)
plt.yticks(ticks=np.arange(max_error//2)*2+1,minor=True)
plt.grid(True,axis="y",which="both")
ax = plt.gca()
ax.grid(which="minor",alpha=0.3,linestyle="--")
plt.ylim((0,max_error))
plt.ylabel("FHR MSE",fontsize="large")
plt.xlabel("Method",fontsize="large")
plt.show()