# CompAugCycleGAN

Source code for [Predicting compositional changes of organic-inorganic hybrid materials with Augmented CycleGAN]().

<img src="https://github.com/qai222/CompAugCycleGAN/blob/main/submission_floats/toc.png" width="643" height="302">

## Python modules
There are three python modules: 
1. [data](./data/) contains classes used for processing amine-templated oxide data.
2. [compemd](./compemd) is used for calculating earth mover distance with user-defined distance matrix. This is based on the [Sinkhorn algorithm implementation](https://github.com/t-vi/pytorch-tvmisc/blob/master/wasserstein-distance/Pytorch_Wasserstein.ipynb) by Thomas Viehmann. 
3. [gans](./gans) is the main module for constructing Augmented CycleGAN. This is based on the [Pytorch code of Augmented CycleGAN](https://github.com/aalmah/augmented_cyclegan) by Amjad Almahairi.

[dataset](./dataset) contains amine-templated oxide datasets used in this study.

[settings.py](./settings.py) contains parameters used in model training. You would also need functions in [utils.py](./utils.py) in dataset generation/training.

## Usage: dataset generation
Create a [dataset](./dataset) folder and simply run [dataset.py](./data/dataset.py) to generate two `.pkl` files: `dataset_ab.pkl` -- chemical compositions, `dimset.pkl` -- dimensionalities.

## Usage: Augmented CycleGAN model training
- The tuned/pretrained models can be downloaded from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5721355.svg)](https://doi.org/10.5281/zenodo.5721355). Extract the three folders in the downloaded `rar` file to [workplace_augcyc/best/](./workplace_augcyc/best/).
- Once the models are placed in [workplace_augcyc/best/](./workplace_augcyc/best/), scripts in [workplace_augcyc/eval](./workplace_augcyc/eval) can be used to evaluate the tuned models.
- Hyperparameter tuning can be done using [tune.py](./workplace_augcyc/gpmin/tune.py). There's also a script there for running on HPC with PBS.

## Usage: dimensionality prediction
- Model tuning is done through the script [workplace_dim/tune_models.py](./workplace_dim/tune_models.py). Evaluations are done by [workplace_dim/tuned.py](./workplace_dim/tuned.py).
- The scripts in [workplace_dim/tandem](./workplace_dim/tandem) are for connecting the composition generators to a dimensionality classifier.
