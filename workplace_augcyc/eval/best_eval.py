import glob
import os.path

import numpy as np
import seaborn as sns

from data import FormulaDataset, GroupAB
from gans.trainer import Trainer
from settings import seed_rng, SEED
from utils import load_pkl
from utils import plot_violin, plot_cdf


def eval_best_aug(dataset, cvfolders):
    prior_data = []
    opt_data = []
    bl_identity = []
    bl_random = []
    for cvfolder in cvfolders:
        print("working on:", cvfolder)
        trainer = Trainer.load(dataset, os.path.join(cvfolder, "Trainer.yml"), change_wdir=cvfolder)
        trainer.load_model()
        trainer.plot_loss()
        prior_ds, prior_ds_names = trainer.eval_model("prior", eval_quantity="mindist", plot=False, ntrials=50, std=1)
        opt_ds, opt_ds_names = trainer.eval_model("opt", eval_quantity="mindist", plot=False, zlim=5, steps=100)

        pd, bli, blr = prior_ds
        od, bli, blr = opt_ds

        prior_data += pd
        opt_data += od
        bl_identity += bli
        bl_random += blr
    return prior_data, opt_data, bl_identity, bl_random,



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    cvfolders = sorted(glob.glob("../best/2021*/"))
    dataset = load_pkl("../../dataset/dataset_ab.pkl")

    dataset.convertmno = False
    dataset: FormulaDataset
    dataset.gab: GroupAB
    seed_rng(SEED)

    sns.set_theme()
    sns.set(font_scale=1.4)
    sns.set_style("whitegrid")

    priordata, optdata, bli, blr = eval_best_aug(dataset, cvfolders)

    for dist in [priordata, optdata, bli, blr]:
        print(np.mean(dist), np.std(dist))
    plot_violin([optdata, priordata, bli], ["opt", "sample", "identity"], ["red", "purple", "gray"], "best_eval_vio.tiff",
                ylim=[-0.005, 0.125])
    plot_cdf([optdata, priordata, bli], ["opt", "sample", "identity", ], ["red", "purple", "gray"], "best_eval_cdf.tiff")
    plot_violin([optdata, priordata, bli], ["opt", "sample", "identity"], ["red", "purple", "gray"], "best_eval_vio.eps",
                ylim=[-0.005, 0.125])
    plot_cdf([optdata, priordata, bli], ["opt", "sample", "identity", ], ["red", "purple", "gray"], "best_eval_cdf.eps")
