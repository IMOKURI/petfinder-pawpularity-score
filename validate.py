import logging
import os

import hydra
import pandas as pd
from omegaconf import OmegaConf

import wandb
from src.get_score import get_result
from src.load_data import load_data

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="main")
def main(c):
    log.info("Started.")

    if c.wandb.enabled:
        c_dict = OmegaConf.to_container(c.validate, resolve=True)
        run = wandb.init(
            entity=c.wandb.entity,
            project=c.wandb.project,
            dir=c.wandb.dir,
            config=c_dict,
            job_type=c.settings.job_type[0],
        )

    train, test, sub = load_data(c)

    for n, base in enumerate(c.validate.base_results):
        oof = pd.read_csv(
            os.path.join(
                c.settings.dirs.working, "..", "base_results", base, "oof_df.csv"
            )
        )
        oof = oof[["Id", "preds"]]
        oof.columns = ["Id", f"preds{n}"]
        train = pd.merge(train, oof, on="Id")

    cols = [f"preds{n}" for n in range(len(c.validate.base_results))]
    train["preds"] = train[cols].values.mean(axis=1)

    score = get_result(c, train, c.params.n_fold)

    train.to_csv("validation_df.csv", index=False)
    if c.wandb.enabled:
        wandb.save("validation_df.csv")

    log.info("Done.")


if __name__ == "__main__":
    main()
