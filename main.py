import hydra
import mlflow

import src.utils as utils


@hydra.main(config_path="config", config_name="main")
def main(config):
    print(config)

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment)

    with mlflow.start_run():
        utils.log_params_from_omegaconf_dict("params", config.params)


if __name__ == "__main__":
    main()
