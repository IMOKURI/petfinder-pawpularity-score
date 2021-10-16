import math
import os
import random
import time

import git
import mlflow
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig


def seed_torch(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def debug_settings(c):
    if c.settings.debug:
        c.settings.print_freq = 10
        c.params.n_fold = 3
        c.params.epoch = 1


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (remain %s)" % (asMinutes(s), asMinutes(rs))


def log_params_from_omegaconf_dict(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            key = f"{k}" if parent_name == "" else f"{parent_name}.{k}"
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                log_params_from_omegaconf_dict(key, v)
            else:
                mlflow.log_param(key, v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            key = f"{i}" if parent_name == "" else f"{parent_name}.{i}"
            mlflow.log_param(f"{parent_name}.{i}", v)


def log_commit_hash():
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    mlflow.set_tag("mlflow.source.git.commit", sha)
