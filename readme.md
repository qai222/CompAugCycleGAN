# CompAugCycleGAN
![Python](https://img.shields.io/badge/Python-3.9-blue.svg?style=for-the-badge&logo=Python&labelColor=E6E600)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10.2-EE4C2C.svg?style=for-the-badge&logo=PyTorch&labelColor=black)
![CUDA](https://img.shields.io/badge/CUDA-11.3-green.svg?style=for-the-badge&logo=NVIDIA&labelColor=gray)
![cc](https://img.shields.io/badge/BY-4.0-black.svg?style=for-the-badge&logo=Creative%20Commons&labelColor=black&color=gray&logoColor=white)
![RSC](https://img.shields.io/badge/RSC-Digital%20Discovery-white?style=for-the-badge&labelColor=DCDCDC&color=05009A&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAzNC4zNiA0MC4xNSI+PHBhdGggZD0iTTI4Mi4yMSw0MDVhMTEuMzMsMTEuMzMsMCwwLDAsMTQuMjYtMS4yLDE5Ljg2LDE5Ljg2LDAsMCwxLDUuMjgsNS4yMiwxOSwxOSwwLDAsMS0yMy42MiwyLjE5Yy0uOC0uNzktMi4xNC0xLjQ0LTIuNzItMi40LDEuNzMtMS42NywzLjQ0LTMuMzksNS4xMS01LjEyQzI4MS4xLDQwNCwyODEuNTcsNDA0LjY2LDI4Mi4yMSw0MDVaIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgtMjY3LjM5IC0zNzQuMTkpIiBzdHlsZT0iZmlsbDojMDIzZDY5Ii8+PHBhdGggZD0iTTI4MC41Miw0MDMuNjhjLTEuNjcsMS43My0zLjM4LDMuNDUtNS4xMSw1LjEyQzI2OSw0MDMsMjY4LjEyLDM5Mi4yNiwyNzMsMzg1LjE2YTYyLjcxLDYyLjcxLDAsMCwxLDYuMTQsNC4xM0MyNzYuMDUsMzkzLjcsMjc2Ljk0LDM5OS44OCwyODAuNTIsNDAzLjY4WiIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTI2Ny4zOSAtMzc0LjE5KSIgc3R5bGU9ImZpbGw6I2YzZWEwMiIvPjxwYXRoIGQ9Ik0yNzkuMTQsMzg5LjI5YTYyLjcxLDYyLjcxLDAsMCwwLTYuMTQtNC4xM2MtNS4yOC0yLjU1LDUuODQtOC43LDguODItOS43LDIuNjgtLjY3LDEyLjQ4LTMuMjUsMTAuNDMsMS43YTIwLjY2LDIwLjY2LDAsMCwwLTIuNjEtLjMxYy00Ljk1LS4yOC0xMS4yOCwxLjY3LTE0LjE0LDUuNjMsMS43NiwxLjM3LDMuNDUsNC4yMSw1LjI4LDQuODVhMTEuMzUsMTEuMzUsMCwwLDEsMTAtMi45Yy0uMjEuNy0uMTEsMS45NC0uNiwyLjM3YTguOTMsOC45MywwLDAsMC02LjU1LDEuNEMyODEuMTYsMzg5Ljc2LDI4Mi4wNywzOTEuNDksMjc5LjE0LDM4OS4yOVoiIHRyYW5zZm9ybT0idHJhbnNsYXRlKC0yNjcuMzkgLTM3NC4xOSkiIHN0eWxlPSJmaWxsOiM5YWQwZTMiLz48cGF0aCBkPSJNMjg5LjY0LDM3Ni44NWExOC42MywxOC42MywwLDAsMSwxMi4wOSw1LjI5Yy0uODEsMS44NC0zLjQ2LDMuNDctNC44Niw1LjIzLTIuMzItMS4zMS01LjA2LTMuNDUtOC4yMy0zLjE2QzI4OS4xMSwzODIuNDMsMjg3LjQ2LDM3Ni43LDI4OS42NCwzNzYuODVaIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgtMjY3LjM5IC0zNzQuMTkpIiBzdHlsZT0iZmlsbDojMDIzZDY5Ii8+PHBhdGggZD0iTTI3MC4xNywzOTEuOTNjLTEuMDYsNiwuNjYsMTIuNzYsNS4yNCwxNi44Ny41OCwxLDEuOTIsMS42MSwyLjcyLDIuNC0yLjE5LDUuNTUtOS4yMi02LjU3LTkuODItOS40MmEyMS40LDIxLjQsMCwwLDEtLjU2LTEwLjE1QzI2OC40NiwzOTEuMzcsMjY5LjM2LDM5MiwyNzAuMTcsMzkxLjkzWiIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTI2Ny4zOSAtMzc0LjE5KSIgc3R5bGU9ImZpbGw6I2Y4ZjM5OSIvPjxwYXRoIGQ9Ik0yODAuNTIsNDAzLjY4YTExLjg2LDExLjg2LDAsMCwxLTMtMTAuMzljLjU1LjUxLDEuOTEuMTIsMi4zNC43MWE5LjQ1LDkuNDUsMCwwLDAsLjcyLDUuMzdjLjkzLDIuNzksNC41NiwyLjg2LDEuNjcsNS42MkMyODEuNTcsNDA0LjY2LDI4MS4xLDQwNCwyODAuNTIsNDAzLjY4WiIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTI2Ny4zOSAtMzc0LjE5KSIgc3R5bGU9ImZpbGw6I2Y4ZjM5OSIvPjxwYXRoIGQ9Ik0yODkuNjQsMzc2Ljg1Yy0yLjE4LS4xNi0uNTMsNS41OC0xLDcuMzYtMy4xNi0uMzMtNiwxLjkyLTguMjgsMy4xMy0xLjQzLTEuNzUtNC0zLjM2LTQuODUtNS4xOUExOC43NSwxOC43NSwwLDAsMSwyODkuNjQsMzc2Ljg1WiIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTI2Ny4zOSAtMzc0LjE5KSIgc3R5bGU9ImZpbGw6IzFlYjBkMSIvPjwvc3ZnPg==)

#### :scroll: Paper: [*Predicting compositional changes of organic-inorganic hybrid materials with Augmented CycleGAN*](https://doi.org/10.1039/D1DD00044F) [![DOI:10.1039/D1DD00044F](https://img.shields.io/badge/DOI-10.1039/D1DD00044F-gray.svg?style=plastic&color=blue&logoColor=white)](https://doi.org/10.1039/D1DD00044F)
#### :arrow_down: Download: [Source code](https://doi.org/10.5281/zenodo.6227643) [![DOI:10.5281/zenodo.6227643](https://img.shields.io/badge/DOI-10.5281/zenodo.6227643-gray.svg?style=plastic&color=blue&logoColor=white)](https://doi.org/10.5281/zenodo.6227643) || [Pretrained](https://doi.org/10.5281/zenodo.5721355)  [![DOI:10.5281/zenodo.5721355](https://img.shields.io/badge/DOI-10.5281/zenodo.5721355-gray.svg?style=plastic&color=blue&logoColor=white)](https://doi.org/10.5281/zenodo.5721355)
#### :clock12: Latest release: [2022/02/22](https://doi.org/10.5281/zenodo.6227643)

## TOC
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
