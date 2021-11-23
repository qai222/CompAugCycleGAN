import os

import numpy as np
import umap

from data import FormulaDataset, GroupAB
from gans.trainer import Trainer
from settings import seed_rng, SEED
from utils import load_pkl

load_folder = "../best/20211013T133839-1/"
load_folder = os.path.abspath(load_folder)


def gen_prior(dataset, nsample):
    trainer = Trainer.load(dataset, os.path.join(load_folder, "Trainer.yml"), change_wdir=load_folder)
    trainer.load_model()
    real_A_encoded, real_B_encoded, fake_A_encoded, fake_B_encoded = trainer.model.predict_from_prior(trainer.dataset,
                                                                                                      n_trials=nsample,
                                                                                                      std=1,
                                                                                                      unique=True,
                                                                                                      back_to_train=False)
    fake_B_encoded_2d = fake_B_encoded.reshape(fake_B_encoded.shape[0] * fake_B_encoded.shape[1],
                                               fake_B_encoded.shape[2])
    return real_B_encoded, fake_B_encoded_2d, real_A_encoded


def compute_emd_distmat(elements, encoded: np.ndarray):
    from compemd.sinkhornOT import element_distance_matrix, ModifiedPettiforScale, chememd_distance_matrix
    from utils import save_pkl
    assert encoded.ndim == 2
    assert len(elements) == encoded.shape[1]
    emd_dist = element_distance_matrix(elements, ModifiedPettiforScale, True)
    chem_emd = chememd_distance_matrix(encoded, emd_dist, 50000)  # exploded for chunk=100k on 2080
    save_pkl(chem_emd, "dimred_emd.pkl")
    return chem_emd


if __name__ == '__main__':
    dataset = load_pkl("../../dataset/dataset_ab.pkl")
    dataset: FormulaDataset
    dataset.gab: GroupAB
    seed_rng(SEED)
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(3.25, 6.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=(1, 1))
    axes = [fig.add_subplot(gs[i]) for i in range(2)]

    real_B_encoded_0, fake_B_encoded_2d_0, real_A_encoded_0 = gen_prior(dataset, nsample=5)

    data = {}
    cats = [real_B_encoded_0, real_A_encoded_0, fake_B_encoded_2d_0]
    all_encoded = np.vstack(cats)

    # l1 dist
    transformer = umap.UMAP(n_neighbors=5, min_dist=0.8, metric="minkowski", metric_kwds={"p": 1}, random_state=SEED)
    x = transformer.fit_transform(all_encoded)

    # # or emd dist
    # from compemd.sinkhornOT import emdloss
    # if os.path.isfile("dimred_emd.pkl"):
    #     emd_distmat = load_pkl("dimred_emd.pkl")
    # else:
    #     emd_distmat = compute_emd_distmat(dataset.gab.possible_elements, all_encoded)
    # transformer = umap.UMAP(n_neighbors=5, min_dist=0.8, metric="precomputed", random_state=SEED)
    # x = transformer.fit_transform(emd_distmat)

    x_0 = x[:len(real_B_encoded_0)]
    x_1 = x[len(x_0): len(x_0) + len(real_A_encoded_0)]
    x_2 = x[len(x_0) + len(x_1): len(x_0) + len(x_1) + len(fake_B_encoded_2d_0)]

    ax0, ax1 = axes

    ax0.scatter(x_0.T[0], x_0.T[1], color="k", alpha=0.8, marker="x", label="Real")
    ax0.scatter(x_1.T[0], x_1.T[1], color="blue", alpha=0.2, marker="*", label="Identity")

    ax1.scatter(x_0.T[0], x_0.T[1], color="k", alpha=0.8, marker="x", label="Real")
    ax1.scatter(x_2.T[0], x_2.T[1], color="purple", alpha=0.2, marker="o", label="Sample")

    labels = ["A", "B", ]
    for ax, label in zip(axes, labels):
        ax.legend(loc="upper left", bbox_to_anchor=[0.01, 0.95], )
        ax.set_xlim([min(x.T[0]) - 0.5, max(x.T[0]) + 0.5])
        ax.set_ylim([min(x.T[1]) - 0.5, max(x.T[1]) + 0.5])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.0, 1.0, label, transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='right')
        ax.axis("off")

    plt.savefig("dimred.tiff", dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
    plt.savefig("dimred.eps", dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
