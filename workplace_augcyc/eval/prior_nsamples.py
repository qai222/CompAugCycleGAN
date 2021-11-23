import glob
import os.path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from data import FormulaDataset, GroupAB
from gans.trainer import Trainer
from settings import seed_rng, SEED
from utils import load_pkl

sns.set_theme()
sns.set_style("whitegrid")


def prior_mean(dataset, nsamples, cvfolders):
    prior_data = []
    bl_identity = []
    bl_random = []
    for cvfolder in cvfolders:
        trainer = Trainer.load(dataset, os.path.join(cvfolder, "Trainer.yml"), change_wdir=cvfolder)
        trainer.load_model()
        prior_ds, prior_ds_names = trainer.eval_model("prior", plot=False, ntrials=nsamples, std=6)
        pd, bli, blr = prior_ds
        prior_data += pd
        bl_identity += bli
        bl_random += blr
    return np.mean(prior_data), np.mean(bl_identity), np.mean(bl_random)


if __name__ == '__main__':
    dataset = load_pkl("../../dataset/dataset_ab.pkl")
    dataset: FormulaDataset
    dataset.gab: GroupAB
    seed_rng(SEED)

    cvfolders = sorted(glob.glob("../best/2021*/"))

    x = []
    y = []
    for nsamples in range(5, 51, 5):
        print("sample", nsamples)
        p, bli, blr = prior_mean(dataset, nsamples, cvfolders)
        x.append(nsamples)
        y.append(p)
    hl_bli = bli
    hl_opt = 0.011604

    plt.plot(x, y, ":o", c="purple", label="sample")
    plt.hlines(hl_bli, xmin=4, xmax=51, label="identity", ls="-", colors="gray", lw=4)
    plt.hlines(hl_opt, xmin=4, xmax=51, label="opt", ls=":", colors="r", lw=4)
    plt.legend(loc="upper right", bbox_to_anchor=(1.0, 0.8))
    plt.ylabel("mean " + r"$\Delta (C, C')$")
    plt.xlabel(r"$N_{\rm{Sample}}$")
    plt.xlim([4, 51])
    plt.savefig("prior_nsamples.tiff", dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
