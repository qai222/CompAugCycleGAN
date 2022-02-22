# CompAugCycleGAN
Latest release: [![DOI](https://zenodo.org/badge/430975947.svg)](https://zenodo.org/badge/latestdoi/430975947)

Source code for **Predicting compositional changes of organic-inorganic hybrid materials with Augmented CycleGAN**.

<img src="https://github.com/qai222/CompAugCycleGAN/blob/main/submission_floats/toc.png" width="643" height="302">

## Python modules
There are three submodules: 
1. [cacgan.data](cacgan/data/) contains classes used for processing amine-templated oxide data. [atmo.csv](cacgan/data/atmo.csv) is the csv form of ATMO dataset.
2. [cacgan.analysis](cacgan/analysis) are functions for analyzing ATMO composition dataset
3. [cacgan.gans](cacgan/gans) is the main module for constructing Augmented CycleGAN. This is based on the [Pytorch code of Augmented CycleGAN](https://github.com/aalmah/augmented_cyclegan) by Amjad Almahairi.

other functions include
- [sinkhornOT.py](cacgan/sinkhornOT.py) is used for calculating earth mover's distance with user-defined distance matrix. 
This is based on the [Sinkhorn algorithm implementation](https://github.com/t-vi/pytorch-tvmisc/blob/master/wasserstein-distance/Pytorch_Wasserstein.ipynb) by Thomas Viehmann. 
- [settings.py](cacgan/settings.py) contains parameters used in model training.

## Scripts
- [generate_dataset.py](scripts/generate_dataset.py) 
create composition datasets used for model training, by default these are 
two `.pkl` files in [dataset](./dataset): `dataset_ab.pkl` -- chemical compositions, `dimset.pkl` -- dimensionalities.
    - additionally, earth mover's distance matrix can be generated using the same script
- [eval_tuned.py](scripts/eval_tuned.py) evaluates the pretrained/tuned model which can be 
downloaded from 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5721355.svg)](https://doi.org/10.5281/zenodo.5721355)
and placed in the [tuned](tuned) folder.
  - alternatively, use `zenodo_get 10.5281/zenodo.5721355` in command line to download
- [hpc_pbs](scripts/hpc_pbs) folder contains code for hyperparameter tuning and submission script for HPC (PBS).
- [tutorial.ipynb](scripts/tutorial.ipynb) a notebook showing how to generate dataset and to train the augmented cycleGAN model.
- [test_emd.py](scripts/test_emd.py) unittest for sinkhorn calculator (testing results against [pyemd](https://github.com/wmayner/pyemd)).
- [test_layer](scripts/test_layer.py) unittest for layers used in models.

## Dimensionality predictor
- Model tuning is done through the script [dimpredictor/tune_models.py](dimpredictor/tune_models.py). Evaluations are done by [dimpredictor/tuned.py](dimpredictor/tuned.py).
- The scripts in [dimpredictor/tandem](dimpredictor/tandem) are for connecting the composition generators to a dimensionality classifier.
