import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error


def get_score(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def get_result(df, fold=None):
    preds = df["preds"].values
    labels = df["Pawpularity"].values
    score = get_score(labels, preds)
    # LOGGER.info(f"Score: {score:<.5f}")
    # if fold == config.n_fold:
    #     wandb.log({"Score": score})
    # else:
    #     wandb.log({f"Score_fold{fold}": score})
    return score
