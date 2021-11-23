import glob
import os.path
import pprint

import matplotlib.pyplot as plt
import seaborn as sns

from data import FormulaDataset, GroupAB
from gans.trainer import Trainer
from settings import seed_rng, SEED
from utils import load_pkl, save_pkl


def eval_best_ratio_aug(dataset, cvfolders):
    reals = []
    rands = []
    fakes = []
    for cvfolder in cvfolders:
        print("working on:", cvfolder)
        trainer = Trainer.load(dataset, os.path.join(cvfolder, "Trainer.yml"), change_wdir=cvfolder)
        trainer.load_model()
        prior_ds, prior_ds_names = trainer.eval_model("prior", eval_quantity="ratio", plot=False, ntrials=200, std=1)
        # opt_ds, opt_ds_names = trainer.eval_model("opt", eval_quantity="ratio", plot=False, zlim=5, steps=100)

        real, rand, fake = prior_ds
        # real, rand, fake = opt_ds
        reals += real.tolist()
        rands += rand.tolist()
        fakes += fake.tolist()

    return reals, rands, fakes


def plot_hist_element_ratio(reals, fakes, saveas="cnratio.tiff"):
    from collections import Counter
    plt.xlim([0, 4.1])
    plt.hist(fakes, bins=400, label=r"$C'_B$", alpha=0.5, facecolor="purple", density=True)
    pprint.pprint({k: v / sum(Counter(reals).values()) for k, v in Counter(reals).items()})

    save_pkl(reals, "cnratio_real.pkl")
    plt.xlabel("C/N ratio")
    plt.ylabel("Probability density")
    plt.legend()
    plt.savefig(saveas, dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
    plt.clf()


if __name__ == '__main__':
    plt.rcParams.update({'font.size': 14})
    cvfolders = sorted(glob.glob("../best/2021*/"))
    dataset = load_pkl("../../dataset/dataset_ab.pkl")

    dataset.convertmno = False
    dataset: FormulaDataset
    dataset.gab: GroupAB
    seed_rng(SEED)

    sns.set_theme()
    sns.set(font_scale=1.4)
    sns.set_style("whitegrid")

    reals, rands, fakes = eval_best_ratio_aug(dataset, cvfolders)

    plot_hist_element_ratio(reals, fakes, "cnratio_hist.tiff")
    plot_hist_element_ratio(reals, fakes, "cnratio_hist.eps")
