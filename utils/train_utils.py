
import torch
import numpy as np
import random

def set_seeds(seed):
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    random.seed(seed)

    return np_rng