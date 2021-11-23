import os
import random

import numpy as np
import torch

GPU_IDs = [0, ]  # on 2080
# GPU_IDs = []  # use cpu

SEED = 42
DEVICE = 'cuda' if len(GPU_IDs) > 0 else 'cpu'
GANS_ROOT = os.path.abspath(os.path.dirname(__file__))

# if use neptune
neptune_api_token = None
neptune_proj_name = None

dim_color_rule = {0: "k", 1: "r", 2: "g", 3: "b", }


def seed_rng(seed=SEED):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
