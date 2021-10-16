import logging
import os

import hydra
import mlflow
import pandas as pd
import timm
import torch

import src.utils as utils
from src.make_fold import make_fold

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="main")
def main(c):
    log.info("Started.")
    os.environ["CUDA_VISIBLE_DEVICES"] = c.settings.gpus

    utils.seed_torch(c.params.seed)
    utils.debug_settings(c)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    ###################################################################################################################
    # Load data
    ###################################################################################################################
    train = pd.read_csv(os.path.join(c.settings.dirs.input, "train.csv"))
    test = pd.read_csv(os.path.join(c.settings.dirs.input, "test.csv"))
    sub = pd.read_csv(os.path.join(c.settings.dirs.input, "sample_submission.csv"))

    train = make_fold(c, train)

    model = timm.create_model(c.params.model_name, pretrained=False)

    mlflow.set_tracking_uri(c.mlflow.tracking_uri)
    mlflow.set_experiment(c.mlflow.experiment)

    with mlflow.start_run():
        utils.log_commit_hash()
        utils.log_params_from_omegaconf_dict("", c.params)

        mlflow.pytorch.log_model(model, c.params.model_name)
        log.info("Done.")
        mlflow.log_artifacts(".")


if __name__ == "__main__":
    main()
