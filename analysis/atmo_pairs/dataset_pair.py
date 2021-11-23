import matplotlib.pyplot as plt
import numpy as np

from data import FormulaDataset, GroupAB, AtmoStructureGroup, AtmoStructure
from utils import load_pkl, composition_elements


def get_chemsys(g: AtmoStructureGroup):
    chemsys = []
    for s in g:
        s: AtmoStructure
        es = composition_elements(s.composition)
        chemsys.append(frozenset(es))
    return chemsys


if __name__ == '__main__':

    dataset = load_pkl("../../dataset/dataset_ab.pkl")
    # dataset = load_pkl("../../datasets/dummy.pkl")
    gab = dataset.gab
    dataset: FormulaDataset
    gab: GroupAB

    # set up figure, 1 column
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(3.8, 2), gridspec_kw={'width_ratios': [1, 1]})

    # panel labels
    ax0.text(-0.35, 1.32, "A", transform=ax0.transAxes,
             fontsize=16, fontweight='bold', va='top', ha='right')
    ax0.text(1.5, 1.32, "B", transform=ax0.transAxes,
             fontsize=16, fontweight='bold', va='top', ha='right')

    # ax0 bar chart for paired/unpaired structures
    xlabels = [
        "{}".format(gab.group_a.first_amine),
        "{}".format(gab.group_b.first_amine),
    ]

    a_unpair = [len(gab.group_a) - len(gab.a_haspair), 0, ]
    a_haspair = [len(gab.a_haspair), 0, ]

    ax0.bar(xlabels, a_unpair, facecolor="none", hatch=None, edgecolor="k", linewidth=2, label="cannot pair")
    ax0.bar(xlabels, a_haspair, bottom=a_unpair, hatch="///", facecolor="none", edgecolor="k", linewidth=2,
            label="can pair")

    b_unpair = [0, len(gab.group_b) - len(gab.b_haspair), ]
    b_haspair = [0, len(gab.b_haspair), ]

    print(len(gab.a_haspair) / len(gab.group_a))
    print(len(gab.b_haspair) / len(gab.group_b))

    ax0.bar(xlabels, b_unpair, facecolor="none", hatch=None, edgecolor="k", linewidth=2)
    ax0.bar(xlabels, b_haspair, bottom=b_unpair, hatch="///", facecolor="none", edgecolor="k", linewidth=2)
    ax0.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.55, 1.28))
    ax0.set_ylim(([0, 500]))
    ax0.set_ylabel("Number of Structures")
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)

    #########################################################
    # ax1 unique chemical systems venn
    a_systems = set(get_chemsys(gab.group_a))
    b_systems = set(get_chemsys(gab.group_b))
    system_sets = (a_systems, b_systems)

    for system in a_systems.union(b_systems):
        print(system, system in a_systems, system in b_systems)

    # v = venn2(subsets = system_sets, set_colors=["white", "white", "none"], set_labels=[gab.group_a.first_amine, gab.group_b.first_amine], alpha=1.0, ax=None)
    alpha = 1.0
    face_colors = ["white", "white", "white"]
    from matplotlib_venn._venn2 import compute_venn2_areas, solve_venn2_circles, compute_venn2_regions, \
        compute_venn2_subsets, prepare_venn_axes

    subsets = system_sets
    if isinstance(subsets, dict):
        subsets = [subsets.get(t, 0) for t in ['10', '01', '11']]
    elif len(subsets) == 2:
        subsets = compute_venn2_subsets(*subsets)
    areas = compute_venn2_areas(subsets, normalize_to=1.0)
    centers, radii = solve_venn2_circles(areas)
    centers = [[centers[0, 1], centers[0, 0]], [centers[1, 1], centers[1, 0]]]
    centers = np.array(centers)
    prepare_venn_axes(ax1, centers, radii)
    regions = compute_venn2_regions(centers, radii)

    patches = [r.make_patch() for r in regions]
    for (p, c) in zip(patches, face_colors):
        if p is not None:
            p.set_facecolor(c)
            p.set_edgecolor('k')
            p.set_alpha(alpha)
            ax1.add_patch(p)
    # ax1.set_aspect("auto")
    print(subsets)
    txts = [gab.group_a.first_amine, gab.group_b.first_amine]
    padding = 0.1
    overlap_pos = np.array([0, 0], dtype=np.float64)
    for i in range(2):
        pos = centers[i]
        # txt = "{}\nexclusive\n{}".format(txts[i], subsets[i])
        txt = "{}\n{}".format(txts[i], subsets[i])
        if pos[1] > 0:
            pos[1] = pos[1] + padding
        else:
            pos[1] = pos[1] - padding
        ax1.text(pos[0], pos[1], txt, size='large', ha='center', va='center')
        overlap_pos += pos
    overlap = patches[-1]
    overlap.set_hatch("///")
    overlap.set_alpha(0.3)
    ax1.text(overlap_pos[0], overlap_pos[1], subsets[-1], size='large', ha='center', va='center')
    ax1.set_title("Chemical Systems", fontsize=11)

    plt.subplots_adjust(
        wspace=0.6,
        # hspace=1
    )

    plt.savefig("dataset_pair.tiff", dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
    plt.savefig("dataset_pair.eps", bbox_inches='tight')
