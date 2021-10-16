import logging
import os

import hydra
import mlflow
import pandas as pd
import torch

import src.utils as utils
from src.get_score import get_result
from src.make_fold import make_fold
from src.train_fold import train_fold

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

    ###################################################################################################################
    # Train
    ###################################################################################################################
    mlflow.set_tracking_uri(c.mlflow.tracking_uri)
    mlflow.set_experiment(c.mlflow.experiment)

    mlflow.start_run()
    utils.log_commit_hash()
    utils.log_params_from_omegaconf_dict("", c.params)

    oof_df = pd.DataFrame()
    for fold in range(c.params.n_fold):
        log.info(f"========== fold {fold} training ==========")
        utils.seed_torch(c.params.seed + fold)

        _oof_df, score, loss = train_fold(c, train, fold, device)
        oof_df = pd.concat([oof_df, _oof_df])

        log.info(f"========== fold {fold} result ==========")
        get_result(_oof_df, fold)

    log.info(f"========== final result ==========")
    get_result(oof_df, c.params.n_fold)

    log.info("Done.")
    mlflow.log_artifacts(".")
    mlflow.end_run()


if __name__ == "__main__":
    main()
