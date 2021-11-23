import datetime
import json
import lzma
import os.path
import pathlib
import pickle
import re
import typing
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from monty.json import MontyDecoder
from pymatgen.core.composition import Composition

_MetalDefined = {"Li", "Be", "Na", "Mg", "Al", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
                 "Ga",
                 "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Cs", "Ba", "La",
                 "Ce",
                 "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re",
                 "Os",
                 "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am",
                 "Cm",
                 "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh",
                 "Fl", "Mc", "Lv"}

_MetalloidDefined = {"Si", "Te", "Sb", "Ge"}

MnM = set.union(_MetalDefined, _MetalloidDefined)


def rearrange_brackets(s: str):  # 2(H2 O1) -> (H2 O1)2
    if "(" not in s:
        return s
    for i, c in enumerate(s):
        if c == "(":
            break
    return s[i:] + s[:i]


def charge_string_to_charge(s: str):
    if "+" in s:
        charge = float(s.replace("+", ""))
    elif "-" in s:
        charge = -float(s.replace("-", ""))
    else:
        raise ValueError("no sign character!")
    return charge


def remove_character(s: str, character="n"):
    nidx = []
    for i, c in enumerate(s):
        if c == character:
            if i == 0:
                nidx.append(i)
            elif s[i - 1].isupper():
                continue
            else:
                nidx.append(i)
    return "".join([s[i] for i in range(len(s)) if i not in nidx])


def csdformula_to_compositions(f: str):
    input_formula = f

    variable_count = 0
    for char in ["n", "x", "y"]:
        before_remove = f
        f = remove_character(f, character=char)
        if f != before_remove:
            variable_count += 1
    if variable_count > 1:
        raise ValueError("there should be only one variable (n, x, y), but we have: {}".format(input_formula))

    fs = f.strip().split(",")

    unit_compositions = []
    unit_charges = []
    multipliers = []
    for formula in fs:
        bra_count = formula.count("(")
        ket_count = formula.count(")")
        assert "[" not in formula and "]" not in formula
        assert "{" not in formula and "}" not in formula
        assert (bra_count, ket_count) in ((0, 0), (1, 1))

        charge_matches = re.findall(r"\d*[+-]{1}", formula)
        assert len(charge_matches) in (0, 1)
        if len(charge_matches) == 0:
            unit_charge = 0
            total_charge = None
        else:
            charge_string = charge_matches[0]
            charge_index = formula.index(charge_string)

            if bra_count == 0 and ket_count == 0:
                unit_charge = charge_string_to_charge(charge_string)
                total_charge = unit_charge
            else:
                if formula.index("(") < charge_index < formula.index(")"):
                    unit_charge = charge_string_to_charge(charge_string)
                    total_charge = None
                else:
                    unit_charge = None
                    total_charge = charge_string_to_charge(charge_string)
        formula = re.sub(r"\d+[+-]", "", formula)

        if bra_count == 0 and ket_count == 0:
            multiplier = 1
            unit_compositions.append(Composition(formula))
            multipliers.append(multiplier)
            unit_charges.append(unit_charge)
        else:
            head = formula[:formula.index("(")].strip()
            tail = formula[formula.index(")") + 1:].strip()
            assert len(head) == 0 or len(tail) == 0
            if len(head) > 0 and len(tail) == 0:
                multiplier = float(head)
            elif len(head) == 0 and len(tail) > 0:
                multiplier = float(tail)
            else:
                multiplier = 1
            if unit_charge is None:
                unit_charge = total_charge / multiplier
            unit_composition = Composition(rearrange_brackets(formula)) / multiplier
            unit_compositions.append(unit_composition)
            multipliers.append(multiplier)
            unit_charges.append(unit_charge)

    return unit_compositions, unit_charges, multipliers


def is_composition_type(c: Composition, composition_type="mo"):
    if composition_type == "mo":
        return c["O"] > 1 and any(c[m] > 0 for m in MnM)
    elif composition_type == "amine":
        return set(c.elements) == set(Composition("CNH").elements)
    elif composition_type == "spacefill":
        return c in [Composition(x) for x in ["H2O", "CH3CH2OH"]]
    else:
        raise NotImplementedError


def load_pkl(fn: typing.Union[str, pathlib.Path], use_lzma=True):
    if use_lzma:
        open = lzma.open
    with open(fn, "rb") as f:
        o = pickle.load(f)
    return o


def save_pkl(o, fn: typing.Union[str, pathlib.Path], use_lzma=True):
    if use_lzma:
        open = lzma.open
    with open(fn, "wb") as f:
        pickle.dump(o, f)


def create_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def diff2formula(f1, f2, norm=True):
    if norm:
        c1 = Composition(f1).fractional_composition
        c2 = Composition(f2).fractional_composition
    else:
        c1 = Composition(f1)
        c2 = Composition(f2)
    e1 = set(c1.elements)
    e2 = set(c2.elements)
    assert e1.issuperset(e2) or e2.issuperset(e1)
    if e1.issuperset(e2):
        se = e1
    else:
        se = e2
    d = 0
    for e in se:
        d += abs(c1[e] - c2[e])
    diff = d / len(se)
    # print(c1, c2, diff)
    return diff


def mindist_ra_fas(ra_e: np.ndarray, fake_A_encoded_2d: np.ndarray, ):
    ra_e_ones = np.zeros(ra_e.shape)
    ra_e_ones[ra_e > 0] = 1
    fake_A_encoded_2d_ones = np.zeros(fake_A_encoded_2d.shape)
    fake_A_encoded_2d_ones[fake_A_encoded_2d > 0] = 1
    diff_ones = fake_A_encoded_2d_ones - ra_e_ones

    row_ids = np.where(~diff_ones.any(axis=1))[0]
    distances = []
    for j in row_ids:
        fa_e = fake_A_encoded_2d[j]
        # assert set(np.nonzero(ra_e)[0]) == set(np.nonzero(fa_e)[0])
        distance = np.abs(ra_e - fa_e)
        distance = distance.sum() / np.count_nonzero(fa_e)
        distances.append(distance)
    return min(distances)


def sns_kde(histdata: dict, saveas, title, xlabel=r"$\Delta$ fraction per element", clip=(0, 0.2), ylabel=None,
            xlim=None, ylim=None, dpi=300):
    fig, ax = plt.subplots()
    for k in histdata:
        v = histdata[k]
        sns.kdeplot(v,
                    clip=clip,
                    ax=ax,
                    label="{} - mean: {:.2e}, std: {:.2e}".format(k, np.mean(v), np.std(v)),
                    # fill=True,
                    # common_norm=False,
                    # palette="crest",
                    # alpha=.2,
                    linewidth=3,
                    )

    ax.set_xlabel(xlabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_yticks([])
    ax.legend()
    ax.set_title(title)
    fig.savefig(saveas, dpi=dpi)


def sns_hist(histdata: dict, saveas, title, xlabel=r"$\Delta$ fraction per element", ylabel=None, xlim=(0, 0.2)):
    fig, ax = plt.subplots()
    for k in histdata:
        v = histdata[k]
        sns.histplot(v,
                     kde=True,
                     kde_kws={"clip": xlim},
                     line_kws={"lw": 3, "ls": "--"},
                     ax=ax,
                     label="{} - mean: {:.2e}, std: {:.2e}".format(k, np.mean(v), np.std(v)),
                     )

    ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_yticks([])
    ax.legend()
    ax.set_title(title)
    fig.savefig(saveas, dpi=300)


def plot_loss(loss_csv: typing.Union[str, pathlib.Path], saveas="loss.png", samewdir=True, title=None, ylim=None):
    df = pd.read_csv(loss_csv)
    fig, ax = plt.subplots()
    x = df["i"]
    for c in df.columns:
        if c not in ["D", "G", "C"]:
            continue
        y = df[c]
        ax.plot(x, y, label=c)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    if ylim:
        ax.set_ylim(ylim)
    ax.legend()
    if title:
        ax.set_title(title)
    if samewdir:
        wdir = os.path.dirname(loss_csv)
        fig.savefig(os.path.join(wdir, saveas))
    else:
        fig.savefig(saveas)


def composition_elements(c: Composition):
    return set(e.name for e in c.elements)


def formula_elements(f: str):
    return set(e.name for e in Composition(f).elements)


def save_yml(o, p: typing.Union[str, pathlib.Path], ):
    with open(p, 'w') as outfile:
        yaml.dump(o, outfile, default_flow_style=True)


def load_yml(p: typing.Union[str, pathlib.Path], ):
    with open(p, 'r') as outfile:
        o = yaml.unsafe_load(outfile)
    return o


def timestamp():
    s = datetime.datetime.now().isoformat()
    return s.replace(":", "").replace("-", "").replace(".", "")[:-6]


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def plot_violin(distributions, names, colors, saveas, ylim=[-0.01, 0.07]):
    records = []
    for i, distribution in enumerate(distributions):
        name = names[i]
        for v in distribution:
            record = {"x": name, "y": v}
            records.append(record)
    df = pd.DataFrame.from_records(records)
    my_pal = dict(zip(names, colors))
    ax = sns.violinplot(
        x="x", y="y",
        # hue="smoker",
        data=df,
        cut=0,  # somewhat ugly...
        palette=my_pal,
    )
    ax.set_ylim(ylim)
    ax.set_xlabel("")
    ax.set_ylabel(r"$\Delta (C, C')$", size=14)
    ax.set_xticklabels(ax.get_xticklabels(), size=14)
    plt.savefig(saveas, dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
    plt.clf()


def plot_cdf(distributions, names, colors, saveas):
    ax = plt.subplot()
    for i, distribution in enumerate(distributions):
        name = names[i]
        distribution = np.array(distribution)
        distribution = distribution[~np.isnan(distribution)]

        count, bins_count = np.histogram(distribution, bins=200)
        pdf = count / sum(count)
        cdf = np.nancumsum(pdf)
        x = bins_count[1:]
        ax.plot(x, cdf, lw=2, label=name, c=colors[i])
        ax.set_ylabel("CDF", size=14)
        ax.set_xlabel(r"$\Delta (C_B, C'_B)$", size=14)
    ax.legend()
    plt.savefig(saveas, dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
    plt.clf()


def get_mno(c: Composition):
    ms = []
    nms = []
    for e in c.elements:
        if c[e] < 1e-5:
            continue
        if e.name in MnM:
            ms.append(e.name)
        elif e.name not in ("C", "N", "H", "O"):
            nms.append(e.name)
    return ms, nms


def convert_mno(encoded_c1, encoded_c2, possible_elements):
    encoded_c1 = encoded_c1.copy()
    encoded_c2 = encoded_c2.copy()
    m_ids, nm_ids, cnho_ids = possible_elements_to_ids(possible_elements)

    nzero_ids1 = np.nonzero(encoded_c1)[0]
    m1_ids = []
    nm1_ids = []
    for i in nzero_ids1:
        if i in m_ids:
            m1_ids.append(i)
        elif i in nm_ids:
            nm1_ids.append(i)

    nzero_ids2 = np.nonzero(encoded_c2)[0]
    m2_ids = []
    nm2_ids = []
    for i in nzero_ids2:
        if i in m_ids:
            m2_ids.append(i)
        elif i in nm_ids:
            nm2_ids.append(i)

    assert len(m1_ids) == len(m2_ids) == 1
    m2id = m2_ids[0]
    m1id = m1_ids[0]
    encoded_c2[m1id] = encoded_c2[m2id]
    encoded_c2[m2id] = 0.0
    assert len(nm1_ids) <= 1
    assert len(nm2_ids) <= 1
    if len(nm2_ids) == 0 and len(nm1_ids) == 0:
        pass
    elif len(nm2_ids) == 0 and len(nm1_ids) == 1:
        pass
    elif len(nm2_ids) == 1 and len(nm1_ids) == 0:
        nm2id = nm2_ids[0]
        encoded_c2[m1id] += encoded_c2[nm2id]
    elif len(nm2_ids) == 1 and len(nm1_ids) == 1:
        nm1id = nm1_ids[0]
        nm2id = nm2_ids[0]
        encoded_c2[nm1id] = encoded_c2[nm2id]
        encoded_c2[nm2id] = 0.0
    return encoded_c2


def possible_elements_to_ids(possible_elements):
    m_ids = []
    nm_ids = []
    cnho_ids = []
    for ie, e in enumerate(possible_elements):
        if e in MnM:
            m_ids.append(ie)
        elif e in ("C", "H", "N", "O"):
            cnho_ids.append(ie)
        else:
            nm_ids.append(ie)
    return m_ids, nm_ids, cnho_ids


def is_mno(c: Composition):
    ms, nms = get_mno(c)
    return len(ms) == 1 and len(nms) == 1


def read_jsonfile(p: typing.Union[str, pathlib.Path], decoder=MontyDecoder):
    with open(p, "r") as f:
        r = json.load(f, cls=decoder)
    return r
