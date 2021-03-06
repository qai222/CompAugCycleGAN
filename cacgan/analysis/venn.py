import itertools
import os.path
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3

from cacgan.data import AtmoStructureGroup, AtmoStructure
from cacgan.utils import read_jsonfile, composition_elements

this_dir = os.path.abspath(os.path.dirname(__file__))
id2year_json = os.path.join(this_dir, "id2year.json")
id2year = read_jsonfile(id2year_json)
years = sorted(set(id2year.values()))

# master_group = AtmoStructureGroup.from_csv()


def popular_amine_groups(whole_group):
    amine_groups = sorted(whole_group.group_by("amine"), key=lambda x: len(x), reverse=True)
    return amine_groups


def master_group_by_year(master_group, year):
    id_discovered = [i for i in id2year if id2year[i] <= year]
    return AtmoStructureGroup([s for s in master_group if s.identifier in id_discovered])


def pies(master_group, kfirst=7):
    popular_amines_over_years = []
    counters = []
    for year in years:
        whole_group = master_group_by_year(master_group, year)
        amine_groups = popular_amine_groups(whole_group)
        popular_amines = [ag.first_amine for ag in amine_groups]
        popular_amines_over_years += popular_amines
        this_year_total_structure = len(whole_group)
        this_year_amine_counter = Counter([s.amine for s in whole_group])
        this_year_popular_amines = sorted(this_year_amine_counter.keys(), key=lambda x: this_year_amine_counter[x],
                                          reverse=True)[:kfirst]
        counters.append(this_year_amine_counter)
    popular_amines_over_years = sorted(set(popular_amines_over_years), key=lambda x: Counter(popular_amines_over_years)[x],
                                    reverse=True)[:kfirst]

    # popular_amines = popular_amines_alltime
    popular_amines = this_year_popular_amines
    for counter, year in zip(counters, years):
        values = [counter[a] for a in popular_amines]

        other_num_structures = sum(counter.values()) - sum(values)
        other_num_amines = max(len(counter.keys()) - len(popular_amines), 0)

        names = popular_amines + ["other"]
        values = values + [other_num_structures]

        patches, texts = plt.pie(values, startangle=90)
        plt.legend(patches, names, bbox_to_anchor=(0.90, 1.025), loc="upper left")
        plt.title("year: {}, total structures: {}, other amines: {}".format(year, sum(counter.values()),
                                                                            other_num_amines), )
        plt.savefig("pies/{}.png".format(year))
        plt.clf()


def venns_one_year(master_group, year=2019, popular_amines=["CNC", "NCCN"]):
    venn_data = []
    nrow = 5
    ncol = 2
    nvenns = 0
    for aminepair in itertools.combinations(popular_amines, 2):
        amine2set = dict()
        whole_group = master_group_by_year(master_group, year)
        info = ""
        for amine in aminepair:
            amine_group = [s for s in whole_group if s.amine == amine]
            chem_syss = []
            for s in amine_group:
                s: AtmoStructure
                chem_syss.append(tuple(sorted(composition_elements(s.composition))))
            chem_syss = set(chem_syss)
            amine2set[amine] = chem_syss
            info += "{}/{}".format(amine, len(chem_syss))
            info += " "
        # print(aminepair, info)
        venn_sets = [amine2set[a] for a in aminepair]
        venn_data.append((aminepair, venn_sets))
        nvenns += 1

    figure, axes = plt.subplots(nrow, nvenns // nrow, figsize=(5, 15))
    for i in range(nvenns):

        ax = axes[i // ncol][i % ncol]
        data = venn_data[i]
        aminepair, venn_sets = data
        if len(venn_sets) == 2:
            vd = venn2(venn_sets, set_labels=aminepair, ax=ax)
            if i % 2 == 1:
                xa, ya = vd.get_label_by_id("A").get_position()
                xb, yb = vd.get_label_by_id("B").get_position()
                # print(ya, yb)
                vd.get_label_by_id("A").set_y(-ya + 0.1)
                vd.get_label_by_id("B").set_y(-yb + 0.1)
        elif len(venn_sets) == 3:
            venn3(venn_sets, set_labels=aminepair, ax=ax)

    #     # ax.set_title("{}".format(year))
    # plt.savefig("analysis_venn_{}_{}.tiff".format("-".join(popular_amines), year), dpi=600, bbox_inches='tight',
    #             pil_kwargs={"compression": "tiff_lzw"})
    # plt.savefig("analysis_venn_{}_{}.eps".format("-".join(popular_amines), year))
    return figure


def venns_last_nyears(
        master_group,
        nyears=30,
        popular_amines=["CNC", "NCCN", "C1CNCCN1"],
):
    venn_data = []
    for year in years[-nyears:]:
        amine2set = dict()
        whole_group = master_group_by_year(master_group, year)
        info = ""
        for amine in popular_amines:
            amine_group = [s for s in whole_group if s.amine == amine]
            chem_syss = []
            for s in amine_group:
                s: AtmoStructure
                chem_syss.append(tuple(sorted(composition_elements(s.composition))))
            chem_syss = set(chem_syss)
            amine2set[amine] = chem_syss
            info += "{}/{}".format(amine, len(chem_syss))
            info += " "
        # print(year, info)

        venn_sets = [amine2set[a] for a in popular_amines]
        venn_data.append((year, venn_sets))

    figure, axes = plt.subplots(5, 6, figsize=(15, 18))

    for i in range(nyears):
        ax = axes[i // 6][i % 6]
        data = venn_data[i]
        year, venn_sets = data
        if len(popular_amines) == 2:
            venn2(venn_sets, set_labels=popular_amines, ax=ax)
        elif len(popular_amines) == 3:
            venn3(venn_sets, set_labels=popular_amines, ax=ax)
        ax.set_title("{}".format(year))
    return figure
    plt.savefig("analysis_venn_{}.tiff".format("-".join(popular_amines)), dpi=600, bbox_inches='tight',
                pil_kwargs={"compression": "tiff_lzw"})
    # plt.savefig("venn_{}.eps".format("-".join(popular_amines)))



