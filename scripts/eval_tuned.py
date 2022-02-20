import glob
import logging
import os.path
import pprint
from collections import Counter

from cacgan.data import FormulaDataset, GroupAB
from cacgan.gans import Trainer
from cacgan.utils import *

"""
evaluate the tuned model
download the tuned model from https://doi.org/10.5281/zenodo.5721355
(you can also use `zenodo_get 10.5281/zenodo.5721355` in cmd to download)
unzip and place the folders at ../tuned/
"""


def plot_bar_element_ratio(real_ratios, saveas="cnratio_barplot.tiff"):
    c = Counter(real_ratios)
    p = {k: v / sum(c.values()) for k, v in c.items()}
    pp = {}
    for k in p:
        near_ps = [kk for kk in pp if abs(kk - k) < 1e-3]
        if len(near_ps) == 0:
            pp[k] = p[k]
        else:
            pp[near_ps[0]] += p[k]
    p = pp
    x = [k for k in p]
    y = [v for v in p.values()]

    f, (ax, ax2) = plt.subplots(2, 1, sharex='all')

    width = 0.05
    ax.bar(x, y, width=width, facecolor="k", label=r"$C_B$")
    ax2.bar(x, y, width=width, facecolor="k")

    ax.set_ylim(.8, 0.86)  # outliers only
    ax2.set_ylim(0, .06)  # most of the data

    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
    plt.xlim([0., 4.1])
    ax2.set_xlabel("C/N ratio")
    ax2.set_ylabel("Probability", loc="top")
    ax2.yaxis.set_label_coords(-0.12, 1.4)
    ax.legend()
    ax.tick_params(length=0.2, top="off", pad=8)
    ax2.tick_params(length=0, pad=8)
    plt.savefig(saveas, dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
    plt.clf()


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


def eval_one_fold(dataset: FormulaDataset, cvfolder: typing.Union[str, pathlib.Path], nsamples: int = 50):
    logging.warning("working on: {}".format(cvfolder))
    trainer = Trainer.load(dataset, os.path.join(cvfolder, "Trainer.yml"), change_wdir=cvfolder)
    trainer.load_model()
    trainer.plot_loss()
    prior_ds, prior_ds_names = trainer.eval_model("prior", eval_quantity="mindist", plot=False, ntrials=nsamples, std=1)
    opt_ds, opt_ds_names = trainer.eval_model("opt", eval_quantity="mindist", plot=False, zlim=5, steps=100)
    pd, bli, blr = prior_ds
    od, bli, blr = opt_ds

    prior_ds_ratio, prior_ds_names_ratio = trainer.eval_model("prior", eval_quantity="ratio", plot=False, ntrials=200,
                                                              std=1)
    ratio_real, ratio_rand, ratio_fake = prior_ds_ratio

    return pd, od, bli, blr, ratio_real, ratio_rand, ratio_fake


def eval_cv(dataset: FormulaDataset, cvfolders: [typing.Union[str, pathlib.Path]]):
    prior_diff = []
    opt_diff = []
    identity_diff = []
    random_diff = []
    ratios_real = []
    ratios_rand = []
    ratios_fake = []
    for cvfolder in cvfolders:
        pd, od, bli, blr, ratio_real, ratio_rand, ratio_fake = eval_one_fold(dataset, cvfolder)
        prior_diff += pd
        opt_diff += od
        identity_diff += bli
        random_diff += blr
        ratios_real += ratio_real.tolist()
        ratios_rand += ratio_rand.tolist()
        ratios_fake += ratio_fake.tolist()
    plot_hist_element_ratio(ratios_real, ratios_fake, "cnratio.tiff")
    plot_bar_element_ratio(ratios_real, "cnratio_barplot.tiff")

    plot_violin([opt_diff, prior_diff, identity_diff], ["opt", "sample", "identity"], ["red", "purple", "gray"],
                "best_eval_vio",
                ylim=[-0.005, 0.125])
    plot_cdf([opt_diff, prior_diff, identity_diff], ["opt", "sample", "identity", ], ["red", "purple", "gray"],
             "best_eval_cdf",
             )


def prior_mean_vs_nsample(dataset, cvfolders, saveas="prior_nsamples.tiff"):
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
    plt.savefig(saveas, dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
    plt.clf()


if __name__ == '__main__':
    dataset = load_pkl("../dataset/dataset_ab.pkl")
    dataset.convertmno = False
    dataset: FormulaDataset
    dataset.gab: GroupAB

    seed_rng(SEED)
    sns.set_theme()
    sns.set(font_scale=1.4)
    sns.set_style("whitegrid")

    tuned_model_path = os.path.abspath("../tuned")
    result_path = os.path.abspath("./eval_tuned/")
    cvfolders = sorted(glob.glob(os.path.join(tuned_model_path, "2021*")))

    os.chdir(result_path)
    eval_cv(dataset, cvfolders)
    # prior_mean_vs_nsample(dataset, cvfolders)
