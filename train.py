import logging
import os

import hydra
import mlflow
import timm
import torch

import src.utils as utils

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="main")
def main(c):
    log.info("Started.")

    utils.seed_torch(c.params.seed)
    utils.debug_settings(c)

    os.environ["CUDA_VISIBLE_DEVICES"] = c.settings.gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

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
