import torch
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

from config.model_config import THRESHOLD


def evaluate_model(
    model,
    dataloader,
    device
):

    model.eval()

    targets_list = []
    predictions_list = []

    with torch.no_grad():

        for features, targets in dataloader:

            features = features.to(device)

            logits = model(features)

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities >= THRESHOLD
            ).int()

            targets_list.append(
                targets.numpy()
            )

            predictions_list.append(
                predictions.cpu().numpy()
            )

    y_true = np.vstack(targets_list)
    y_pred = np.vstack(predictions_list)

    precision = precision_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0
    )

    return precision, recall, f1