import logging
import time

import numpy as np
import torch.cuda.amp as amp

from get_score import get_score
from make_dataset import make_dataloader, make_dataset
from make_loss import make_criterion, make_optimizer, make_scheduler
from make_model import make_model
from train_epoch import train_epoch, validate_epoch
from utils import EarlyStopping

log = logging.getLogger("__main__").getChild("train_loop")


def train_loop(c, df, fold, device):
    # ====================================================
    # Data Loader
    # ====================================================
    trn_idx = df[df["fold"] != fold].index
    val_idx = df[df["fold"] == fold].index

    train_folds = df.loc[trn_idx].reset_index(drop=True)
    valid_folds = df.loc[val_idx].reset_index(drop=True)

    train_ds = make_dataset(c, train_folds, "train")
    valid_ds = make_dataset(c, valid_folds, "valid")

    train_loader = make_dataloader(c, train_ds, shuffle=True, drop_last=True)
    valid_loader = make_dataloader(c, valid_ds, shuffle=False, drop_last=False)

    # ====================================================
    # Model
    # ====================================================
    model = make_model(c)
    model.to(device)

    criterion = make_criterion(c)
    optimizer = make_optimizer(c, model)
    scaler = amp.GradScaler(enabled=c.settings.amp)
    scheduler = make_scheduler(c, optimizer, train_ds)

    es = EarlyStopping(
        patience=c.params.es_patience,
        verbose=True,
        path=f"{c.params.model_name.replace('/', '-')}_fold{fold}",
    )

    # ====================================================
    # Loop
    # ====================================================
    for epoch in range(c.params.epoch):
        start_time = time.time()

        # train
        avg_loss = train_epoch(
            c,
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            scaler,
            epoch,
            device,
        )

        # eval
        avg_val_loss, preds = validate_epoch(c, valid_loader, model, criterion, device)
        valid_labels = valid_folds["Pawpularity"].values

        if c.params.criterion == "BCEWithLogitsLoss":
            preds = 1 / (1 + np.exp(-preds))

        preds *= 100.0

        # scoring
        # score = get_score(valid_labels, preds.argmax(1))
        score = get_score(valid_labels, preds)

        elapsed = time.time() - start_time
        log.info(
            f"Epoch {epoch+1} - "
            f"train_loss: {avg_loss:.4f} "
            f"val_loss: {avg_val_loss:.4f} "
            f"score: {score} "
            f"time: {elapsed:.0f}s"
        )

        es(avg_val_loss, score, model, preds)

        if es.early_stop:
            log.info("Early stopping")
            break

    # valid_folds[[str(c) for c in range(c.settings.n_class)]] = es.best_preds
    # valid_folds["preds"] = es.best_preds.argmax(1)
    valid_folds["preds"] = es.best_preds

    return valid_folds, es.best_score, es.best_loss
