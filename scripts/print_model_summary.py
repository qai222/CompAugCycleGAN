import os

from cacgan.data import FormulaDataset, GroupAB
from cacgan.gans import AugCycleGan
from cacgan.gans import Trainer
from cacgan.utils import SEED, seed_rng, load_pkl

"""
print a txt file describing the structure of augcyc
"""

seed_rng(SEED)

dataset = load_pkl("../dataset/dataset_ab.pkl")
dataset: FormulaDataset
dataset.gab: GroupAB

trainer_dir = os.path.abspath("../tuned/20211013T133839-1")
trainer = Trainer.load(dataset, "{}/Trainer.yml".format(trainer_dir), change_wdir=trainer_dir)
trainer.model: AugCycleGan
trainer.model.write_structure(outpath=os.path.abspath("model_summary.txt"))
