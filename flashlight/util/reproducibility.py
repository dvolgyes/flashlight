#!/usr/bin/env python3
import numpy as np
import torch
import time
from .gittools import git_summary


def ensure_reproducible_results(cfg):
    seed = 0
    if 'initial_seed' in cfg:
        seed = cfg.initial_seed
    else:
        seed = int(time.time() * 1000000) % (2 ** 32 - 1)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg.initial_seed = seed
    sha, *_ = git_summary()
    cfg.git_hash = sha[:8]
    return cfg
