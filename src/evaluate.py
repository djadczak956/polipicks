import torch
import numpy as np

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

from config.model_config import THRESHOLD_GRID


def predict_probabilities(
    model,
    dataloader,
    device
):

    model.eval()

    targets_list = []
    probabilities_list = []

    with torch.no_grad():

        for features, targets in dataloader:

            features = features.to(device)

            logits = model(features)

            probabilities = torch.sigmoid(
                logits
            )

            targets_list.append(
                targets.numpy()
            )

            probabilities_list.append(
                probabilities.cpu().numpy()
            )

    y_true = np.vstack(targets_list)
    probabilities = np.vstack(probabilities_list)

    return y_true, probabilities


def score_predictions(y_true, y_pred):

    # Micro: every member-sector cell counts equally.
    # Macro: average over sectors, so rare sectors count as much as common ones.
    # Accuracy is left out on purpose: predicting "no trade" everywhere
    # is ~96% accurate.
    return {
        "precision": precision_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
    }


def choose_threshold(y_true, probabilities):

    # Pick the threshold with the best micro F1. Only call this on
    # tuning data, never on the test period.
    f1_scores = [
        f1_score(
            y_true,
            probabilities >= threshold,
            average="micro",
            zero_division=0
        )
        for threshold in THRESHOLD_GRID
    ]

    return float(
        THRESHOLD_GRID[int(np.argmax(f1_scores))]
    )


def evaluate_model(
    model,
    dataloader,
    device,
    threshold
):

    y_true, probabilities = predict_probabilities(
        model,
        dataloader,
        device
    )

    y_pred = (
        probabilities >= threshold
    ).astype(int)

    return score_predictions(
        y_true,
        y_pred
    )
