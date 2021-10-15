import os
import random

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
