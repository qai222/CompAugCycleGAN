import os

import matplotlib.pyplot as plt
import numpy as np

from data.dataset import FormulaDataset
from gans.trainer import Trainer
from settings import SEED, seed_rng
from utils import load_pkl, MnM

seed_rng(SEED)

cm = plt.cm.get_cmap('cool')

dataset = load_pkl("../../dataset/dataset_ab.pkl")
dataset: FormulaDataset
trainer_dir = os.path.abspath("../../workplace_augcyc/best/20211013T133839-2/")
trainer = Trainer.load(dataset, "{}/Trainer.yml".format(trainer_dir), change_wdir=trainer_dir)
trainer.load_model()
dataset = trainer.dataset  # setup is done at this point
td = dataset.as_test()

import seaborn as sns

sns.set_theme()
sns.set_style("whitegrid")


def gen_prior(nsample):
    # trainer = Trainer.load(dataset, os.path.join(load_folder, "Trainer.yml"), change_wdir=load_folder)
    real_A_encoded, real_B_encoded, fake_A_encoded, fake_B_encoded = trainer.model.predict_from_prior(trainer.dataset,
                                                                                                      n_trials=nsample,
                                                                                                      std=1,
                                                                                                      unique=True,
                                                                                                      back_to_train=False)
    fake_B_encoded_2d = fake_B_encoded.reshape(fake_B_encoded.shape[0] * fake_B_encoded.shape[1],
                                               fake_B_encoded.shape[2])
    return real_A_encoded, real_B_encoded, fake_B_encoded_2d, fake_B_encoded


def fill_zeros(composition_vect, elements1: list, elements2: list):
    assert len(composition_vect) == len(elements1)
    assert set(elements2).issuperset(set(elements1))
    new_vect = np.zeros(len(elements2))
    for i in range(len(elements2)):
        if elements2[i] in elements1:
            new_vect[i] = composition_vect[elements1.index(elements2[i])]
    return new_vect


def get_element_ratio(v, elements, e1, e2):
    ie1 = elements.index(e1)
    ie2 = elements.index(e2)
    return v[ie1] / v[ie2]


def get_mo_ratio(v, elements):
    mor = 0.0
    for e in elements:
        if e in MnM:
            mor += get_element_ratio(v, elements, e1=e, e2="O")
    return mor


def fill_zeros_2d(a2d, elements1, elements2):
    a = np.zeros((a2d.shape[0], len(elements2)))
    for i in range(a2d.shape[0]):
        a[i] = fill_zeros(a2d[i], elements1, elements2)
    return a


def get_chemsys(v, elements):
    nzi = [i for i in range(len(v)) if v[i] != 0]
    return tuple(sorted([elements[i] for i in nzi]))


def get_vs_by_chemsys(a2d, elements, chemsys):
    a = []
    iis = []
    for i in range(len(a2d)):
        csys = get_chemsys(a2d[i], elements)
        if csys == chemsys:
            a.append(a2d[i])
            iis.append(i)
    return np.array(a), iis


if __name__ == '__main__':

    realas, realbs, fakebs, fakebs_folded = gen_prior(nsample=50)

    elements1 = dataset.gab.possible_elements

    maxx = 0
    maxy = 0
    add_legend = True
    for i in range(realbs.shape[0]):
        real_b = realbs[i]
        csys = get_chemsys(real_b, elements1)
        if csys != ("Al", "C", "H", "N", "O", "P"):
            continue
        real_a = realas[i]
        fakebs_from_this_real_b = fakebs_folded[:, i, :]

        x = get_mo_ratio(real_a, elements1)
        ys = [get_mo_ratio(v, elements1) for v in fakebs_from_this_real_b]
        yys = get_mo_ratio(real_b, elements1)
        xs = [x] * len(ys)
        if max(xs) > maxx:
            maxx = max(xs)
        if max(ys) > maxy:
            maxy = max(ys)
        if add_legend:
            plt.scatter(xs, ys, marker="o", c="b", alpha=0.1, label=r"$C'_B$")
            plt.hlines(yys, -1, 1, colors="k", ls=":", label=r"$C_B$", alpha=1.0, zorder=1000)
            add_legend = False
        else:
            plt.scatter(xs, ys, marker="o", c="b", alpha=0.1)
            plt.hlines(yys, -1, 1, colors="k", ls=":", alpha=1.0, zorder=1000)

    plt.xlabel(r"Al:O in $C_A$")
    plt.ylabel(r"Al:O in $C_B$ or $C'_B$")
    xs = np.linspace(0, 1, 100)
    plt.axis('scaled')
    plt.legend(loc="upper left")
    plt.xlim([-0.02, 0.3])
    plt.ylim([-0.02, 0.3])
    plt.savefig("ratio_gen.tiff", dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
