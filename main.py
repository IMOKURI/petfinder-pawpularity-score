import logging

import hydra
import mlflow

import src.utils as utils

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="main")
def main(c):
    log.info("Started.")

    mlflow.set_tracking_uri(c.mlflow.tracking_uri)
    mlflow.set_experiment(c.mlflow.experiment)

    with mlflow.start_run():
        utils.log_params_from_omegaconf_dict("params", c.params)

        log.info("Done.")
        mlflow.log_artifacts(c.settings.output_dir)



if __name__ == "__main__":
    main()
