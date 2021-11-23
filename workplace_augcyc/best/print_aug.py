import os

from data import FormulaDataset, GroupAB
from gans.augcyc import AugCycleGan
from gans.trainer import Trainer
from settings import SEED, seed_rng
from utils import load_pkl

seed_rng(SEED)

dataset = load_pkl("../../dataset/dataset_ab.pkl")
dataset: FormulaDataset
dataset.gab: GroupAB

trainer_dir = os.path.abspath("20211013T133839-1")
trainer = Trainer.load(dataset, "{}/Trainer.yml".format(trainer_dir), change_wdir=trainer_dir)
trainer.model: AugCycleGan
trainer.model.write_structure()
