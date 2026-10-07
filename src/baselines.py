"""Baselines scored on the same walk-forward folds as main.py.

The neural net's numbers only mean something next to these:
  - repeat:   predict the sectors the member traded (disclosed) in the last 90 days
  - logistic: one logistic regression per sector
  - boosting: one gradient-boosted tree model per sector

Logistic and boosting pick their threshold the same way as main.py: on the
last TUNING_DAYS of the training period, never on the test period.

Run: python -m src.baselines   (or: make baselines)
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config.model_config import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    SECTORS,
    DATE_COLUMN,
    SEED,
)
from src.evaluate import choose_threshold, score_predictions
from src.splits import walk_forward_splits, split_for_tuning


MODELS = {
    "logistic": lambda: LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
    ),
    "boosting": lambda: HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        random_state=SEED,
    ),
}


def fit_and_predict(make_model, train_df, eval_df):

    scaler = StandardScaler().fit(train_df[FEATURE_COLUMNS])
    x_train = scaler.transform(train_df[FEATURE_COLUMNS])
    x_eval = scaler.transform(eval_df[FEATURE_COLUMNS])

    probabilities = np.zeros((len(eval_df), len(TARGET_COLUMNS)))

    for i, target in enumerate(TARGET_COLUMNS):

        y = train_df[target]

        # A sector with no positives in training can't be fit; predict 0.
        if y.nunique() < 2:
            continue

        model = make_model().fit(x_train, y)
        probabilities[:, i] = model.predict_proba(x_eval)[:, 1]

    return probabilities


def main():

    df = pd.read_parquet("data/processed/model_data.parquet")
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

    rows = []

    for fold_number, fold in enumerate(walk_forward_splits(df), start=1):

        train_df, test_df = fold["train"], fold["validation"]

        if train_df.empty or test_df.empty:
            continue

        y_test = test_df[TARGET_COLUMNS].values

        repeat = (
            test_df[[f"{s}_trades_90d" for s in SECTORS]].values > 0
        ).astype(int)

        rows.append({
            "fold": fold_number,
            "model": "repeat last 90d",
            **score_predictions(y_test, repeat),
        })

        fit_df, tune_df = split_for_tuning(train_df)

        for name, make_model in MODELS.items():

            threshold = choose_threshold(
                tune_df[TARGET_COLUMNS].values,
                fit_and_predict(make_model, fit_df, tune_df),
            )

            p_test = fit_and_predict(make_model, train_df, test_df)

            rows.append({
                "fold": fold_number,
                "model": name,
                **score_predictions(y_test, (p_test >= threshold).astype(int)),
            })

            print(f"fold {fold_number} {name} done", flush=True)

    results = pd.DataFrame(rows)

    print("\nPer fold (f1):")
    print(
        results.pivot(index="fold", columns="model", values="f1")
        .round(4)
        .to_string()
    )

    print("\nMean over folds:")
    print(
        results.drop(columns="fold")
        .groupby("model", sort=False)
        .mean()
        .round(4)
        .to_string()
    )


if __name__ == "__main__":
    main()
