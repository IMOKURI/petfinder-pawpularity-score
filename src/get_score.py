import logging

import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error

log = logging.getLogger("__main__").getChild("get_score")


def get_score(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def get_result(df, fold):
    preds = df["preds"].values
    labels = df["Pawpularity"].values
    score = get_score(labels, preds)
    log.info(f"Score: {score:<.5f}")
    mlflow.log_metric("score", score, fold)
    return score
