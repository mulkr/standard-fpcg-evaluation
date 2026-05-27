# Standardized fPCG evaluation
This is the code used for our article: An End-to-End Framework for Fetal Phonocardiography Processing Method Evaluation and Comparison

Citation: (coming soon)

## Benchmark usage
Recommended Python version: 3.11

> [!IMPORTANT]
> Make sure to use the correct data and result paths. These are currently "hardcoded" to use my setup
> If you encounter problems with installing the requirements, let me know. Or open an issue/PR.

* Install the dependencies (preferably in a virtual environment)
* Run the detection algorithm(s) to generate detection files
```
  python methods/<method>.py <input dir> <output dir>
```
* Make the appropriate changes to a benchmark runner (e.g. `run_test_benchmark.py`)
* OR: Create your own benchmark based on the provided benchmark runners
* Run your benchmark

### Matlab sources
Certain methods were originally implemented in Matlab. These are redistributed under the GPLv3 licence. Several small changes were made, these are usually marked with a comment. The detection file generation for these is also written in Matlab. Our original code includes the following files: `convert_segments.m`, `corssvalid_Schmidt.m`, `crossvalid_Springer.m`, `test_Schmidt_pretrained.m`, `test_Springer_pretrained.m`, `train_test_Schmidt.m`, `train_test_Springer.m`

### Renna method
The following implementation was used for training: [https://github.com/eneriz-daniel/PCG-Segmentation-Model-Optimization](https://github.com/eneriz-daniel/PCG-Segmentation-Model-Optimization).

Relevant parts of the code was extracted for inference. This is located in `methods/renna/preprocess.py`. Pretrained models are also included.

## Data Generator usage
To generate the same data used in the article run `dataset_gen.py` and `noise_levels_dataset_gen.py`

This dataset is also available to download: [https://zenodo.org/records/20269659](https://zenodo.org/records/20269659)

### Custom generation
A synthetic signal is parametrized with four "parameter groups". These encode the fetal and maternal heart rate and heart sound properties, the specific noise levels, and fetal heart rate modifiers (accelerations, decelerations).

For further details see the provided examples.

## Correspondence
Kristóf Müller: muller.kristof@itk.ppke.hu
