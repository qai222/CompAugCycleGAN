import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.ensemble import RandomForestClassifier
from skopt import BayesSearchCV
from skopt import load

from data.dataset import DimDataset, FormulaDataset
from gans.trainer import Trainer
from settings import SEED, dim_color_rule, seed_rng
from utils import load_pkl, MnM

seed_rng(SEED)

cm = plt.cm.get_cmap('cool')
transformer = umap.UMAP(n_neighbors=20, min_dist=0.5, metric="minkowski", metric_kwds={"p": 1}, random_state=SEED)

dataset = load_pkl("../../dataset/dataset_ab.pkl")
dimset = load_pkl("../../dataset/dimset.pkl")
dimset: DimDataset
dataset: FormulaDataset

trainer_dir = os.path.abspath("../../workplace_augcyc/best/20211013T133839-2/")
trainer = Trainer.load(dataset, "{}/Trainer.yml".format(trainer_dir), change_wdir=trainer_dir)
trainer.load_model()
dataset = trainer.dataset  # setup is done at this point

test_bpool = sorted(set(dataset.test_bpool))
train_bpool = sorted(set(dataset.train_bpool))
test_bs = [dataset.gab.group_b[i].identifier for i in test_bpool]
train_bs = [dataset.gab.group_b[i].identifier for i in train_bpool]
dimset.holdout(test_bs)

rf_opt = load("../tune_models/opt-rf.pkl")
rf_opt: BayesSearchCV
rf_clf = RandomForestClassifier(random_state=SEED, **rf_opt.best_params_)
rf_clf.fit(dimset.x.values, dimset.y.values.ravel(), )
s = rf_clf.score(dimset.hx.values, dimset.hy.values.ravel())
print(s)


def gen_prior(nsample):
    # trainer = Trainer.load(dataset, os.path.join(load_folder, "Trainer.yml"), change_wdir=load_folder)
    real_A_encoded, real_B_encoded, fake_A_encoded, fake_B_encoded = trainer.model.predict_from_prior(trainer.dataset,
                                                                                                      n_trials=nsample,
                                                                                                      std=1,
                                                                                                      unique=True,
                                                                                                      back_to_train=False)
    fake_B_encoded_2d = fake_B_encoded.reshape(fake_B_encoded.shape[0] * fake_B_encoded.shape[1],
                                               fake_B_encoded.shape[2])
    return real_B_encoded, fake_B_encoded_2d


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

    clf = rf_clf

    realbs, fakebs = gen_prior(nsample=100)

    elements1 = dataset.gab.possible_elements
    elements2 = dimset.x.columns.tolist()

    fakebs = fill_zeros_2d(fakebs, elements1, elements2)

    test_bpool = sorted(set(dataset.test_bpool))
    test_bs = [dataset.gab.group_b[i].identifier for i in test_bpool]
    realbs_sys_counter = Counter([get_chemsys(v, elements1) for v in realbs])

    total_hit = 0
    total_mis = 0
    for this_chemsys in realbs_sys_counter:
        if this_chemsys != ("Al", "C", "H", "N", "O", "P"):
            continue

        fakebs_chemsys, _ = get_vs_by_chemsys(fakebs, elements2, this_chemsys)

        x_test, x_test_ii = get_vs_by_chemsys(dimset.hx.values, elements2, this_chemsys)
        y_test = [dimset.hy.values.ravel()[i] for i in x_test_ii]
        y_test_pred = clf.predict(x_test)

        x_train, x_train_ii = get_vs_by_chemsys(dimset.x.loc[train_bs].values, elements2, this_chemsys)
        y_train = [dimset.y.values.ravel()[i] for i in x_train_ii]
        if len(x_train) == 0:
            y_train_pred = []
        else:
            y_train_pred = clf.predict(x_train)

        x_from_gan = fakebs_chemsys

        y_from_gan = clf.predict(x_from_gan)

        hit = 0
        mis = 0
        for ix, x_v in enumerate(x_test):
            l1_dists = np.sum(np.abs(x_from_gan - x_v), axis=1)
            minind = np.argmin(l1_dists)
            if y_test[ix] == y_from_gan[minind]:
                hit += 1
            else:
                mis += 1
        total_hit += hit
        total_mis += mis

        if len(x_test) >= 4:
            print("work on:", this_chemsys)
            print("gen comps", x_from_gan.shape)
            print("test bpair", x_test.shape)
            print("train bpair", x_train.shape)
            if len(x_train) == 0:
                all_encoded = np.vstack([x_test, x_from_gan])
            else:
                all_encoded = np.vstack([x_test, x_from_gan, x_train])
            x = transformer.fit_transform(all_encoded)

            p_test_x = x[:len(x_test)].T[0]
            p_test_y = x[:len(x_test)].T[1]
            p_gan_x = x[len(x_test):len(x_test) + len(x_from_gan)].T[0]
            p_gan_y = x[len(x_test):len(x_test) + len(x_from_gan)].T[1]
            p_train_x = x[len(x_test) + len(x_from_gan):].T[0]
            p_train_y = x[len(x_test) + len(x_from_gan):].T[1]

            xmin = min(x.T[0])
            xmax = max(x.T[0])
            ymin = min(x.T[1])
            ymax = max(x.T[1])

            colors = list(dim_color_rule.values())
            pts_gan = list(zip(p_gan_x, p_gan_y, [dim_color_rule[d] for d in y_from_gan], x_from_gan, y_from_gan))

            # # real dim
            # pts_test = list(zip(p_test_x, p_test_y, [dim_color_rule[d] for d in y_test]))
            # pts_train = list(zip(p_train_x, p_train_y, [dim_color_rule[d] for d in y_train]))
            # pred dim
            pts_test = list(zip(p_test_x, p_test_y, [dim_color_rule[d] for d in y_test_pred]))
            pts_train = list(zip(p_train_x, p_train_y, [dim_color_rule[d] for d in y_train_pred]))
            print("test y", Counter(y_test_pred))
            print("train y", Counter(y_train_pred))
            print("generated y", Counter(y_from_gan))

            for dim in dim_color_rule:
                color = dim_color_rule[dim]

                # plt test dim
                pts_c = [pt for pt in pts_test if pt[2] == color]
                xx = [pt[0] for pt in pts_c]
                yy = [pt[1] for pt in pts_c]
                c = [pt[2] for pt in pts_c]
                if len(xx) == 0:
                    plt.scatter(xx, yy, c=c, marker="^")
                else:
                    plt.scatter(xx, yy, c=c, marker="^", label="Real {}D".format(dim))

                # plot gan gen dim
                pts_c = [pt for pt in pts_gan if pt[2] == color]
                xx = [pt[0] for pt in pts_c]
                yy = [pt[1] for pt in pts_c]
                c = [pt[2] for pt in pts_c]
                if len(xx) == 0:
                    plt.scatter(xx, yy, c=c, marker="o", alpha=0.1)
                else:
                    plt.scatter(xx, yy, c=c, marker="o", label="Generated {}D".format(dim), alpha=0.1)

                # plot training dim
                pts_c = [pt for pt in pts_train if pt[2] == color]
                xx = [pt[0] for pt in pts_c]
                yy = [pt[1] for pt in pts_c]
                c = [pt[2] for pt in pts_c]
                if len(xx) == 0:
                    plt.scatter(xx, yy, c=c, marker="s")
                else:
                    plt.scatter(xx, yy, c=c, marker="s", label="Real {}D (train)".format(dim))

            plt.axis("off")
            plt.title("Chemical System: {}".format(" ".join(this_chemsys)))
            plt.legend(ncol=3, loc="lower center", bbox_to_anchor=[0.5, -0.2])
            plt.tight_layout()
            plt.savefig("dimred-{}.tiff".format("".join(this_chemsys)), dpi=600, bbox_inches='tight',
                        pil_kwargs={"compression": "tiff_lzw"})
            plt.savefig("dimred-{}.eps".format("".join(this_chemsys)), dpi=600, bbox_inches='tight',
                        pil_kwargs={"compression": "tiff_lzw"})

            plt.cla()
            # plot gan gen ratio
            xx = [pt[0] for pt in pts_gan]
            yy = [pt[1] for pt in pts_gan]
            c = [get_mo_ratio(pt[3], elements2) for pt in pts_gan]
            if len(xx) == 0:
                sc = plt.scatter(xx, yy, c=c, marker="o", alpha=0.1, cmap=cm)
            else:
                sc = plt.scatter(xx, yy, c=c, marker="o", label="ratio", alpha=0.1, cmap=cm)
            plt.axis("off")
            plt.title("Chemical System: {}".format(" ".join(this_chemsys)))
            plt.xlim([xmin - 0.5, xmax + 0.5])
            plt.ylim([ymin - 0.5, ymax + 0.5])
            plt.colorbar(sc, label="M/O ratio")
            plt.savefig("dimred-{}-ratio.tiff".format("".join(this_chemsys)), dpi=600, bbox_inches='tight',
                        pil_kwargs={"compression": "tiff_lzw"})
            plt.clf()

            # break

    print(total_hit, total_mis, total_hit / (total_mis + total_hit))
