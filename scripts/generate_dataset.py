import logging

from cacgan.data import AtmoStructureGroup, GroupAB, FormulaDataset, DimDataset, FormulaEncoder
from cacgan.utils import *

"""
script to generate datasets, by default it will put datasets at ../dataset/
"""

DataSetDir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../dataset/")
create_dir(DataSetDir)


def generate_emd(save_as: typing.Union[str, pathlib.Path] = os.path.join(DataSetDir, "emd_data.pkl")) -> dict:
    """earth mover's distance matrix for the master atmo group"""
    from cacgan.sinkhornOT import ModifiedPettiforScale, element_distance_matrix, chememd_distance_matrix
    scale_used = ModifiedPettiforScale
    master_group = AtmoStructureGroup.from_csv()
    fe = FormulaEncoder(master_group.possible_elements)
    encoded = fe.encode_2d([s.composition.fractional_composition.formula for s in master_group])
    emd_dist = element_distance_matrix(master_group.possible_elements, scale_used, True)
    chem_emd = chememd_distance_matrix(encoded, emd_dist, 50000)  # exploded for chunk=80k on rtx 2080...
    data = {
        "master_group": master_group,
        "emd_distance_matrix": chem_emd,
        "encoded": encoded,
        "scale": scale_used,
    }
    save_pkl(data, save_as)
    logging.warning("emd dataset saved at: {}".format(save_as))
    return data


def generate_comp(
        formula_ds=os.path.join(DataSetDir, "dataset_ab.pkl"),
        dim_ds=os.path.join(DataSetDir, "dimset.pkl"),
        # formula_mo_ds=os.path.join(DataSetDir, "dataset_ab_mo.pkl"),
        # dim_feat_ds = os.path.join(DataSetDir, "dimset_feat.pkl"),
):

    # a group pair consists of the 2 most popular amine groups
    gb, ga = sorted(AtmoStructureGroup.from_csv().group_by("amine"), key=lambda x: len(x), reverse=True)[:2]
    gab = GroupAB(ga, gb)
    dataset_ab = FormulaDataset(gab, mo=False, mno=False)
    save_pkl(dataset_ab, formula_ds)
    logging.warning("formula dataset saved at: {}".format(formula_ds))

    # or mo only composition
    # dataset_ab = FormulaDataset(gab, mo=True, mno=False)
    # save_pkl(dataset_ab, formula_mo_ds)

    # save dim dataset without matminer feats
    master_group = AtmoStructureGroup.from_csv()
    dd = DimDataset.from_atmogroup(master_group, feat=False, mocomp=False)
    save_pkl(dd, dim_ds)
    logging.warning("dimensionality dataset saved at: {}".format(dim_ds))

    # # or with feats
    # dd = DimDataset.from_atmogroup(master_group, feat=True, mocomp=False)
    # save_pkl(dd, dim_feat_ds)


if __name__ == '__main__':
    seed_rng(SEED)

    # generate composition vector dataset
    generate_comp()

    # # generate emd matrix between composition vectors
    # generate_emd()
