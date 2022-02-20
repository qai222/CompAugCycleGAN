from cacgan.gans import tune
from cacgan.data import FormulaDataset, GroupAB
from cacgan.utils import load_pkl, seed_rng, SEED


if __name__ == '__main__':
    seed_rng(SEED)
    dataset = load_pkl("../../dataset/dataset_ab.pkl")
    dataset: FormulaDataset
    gab = dataset.gab
    gab: GroupAB

    chk = None
    # from skopt import load
    # chk = load("checkpoint.pkl")  # restart
    tune(dataset, chk=chk, nepochs=10002, n_calls=50)
