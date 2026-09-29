import pandas as pd

from config.model_config import (
    PREDICTION_WINDOW_DAYS,
    DATE_COLUMN
)


def walk_forward_splits(df):

    df = df.copy()

    df[DATE_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN]
    )

    validation_periods = [
        ("2023-01-01", "2023-12-31"),
        ("2024-01-01", "2024-12-31"),
        ("2025-01-01", "2025-12-31"),
        ("2026-01-01", "2026-03-31"),
    ]

    folds = []

    for val_start, val_end in validation_periods:

        val_start = pd.Timestamp(val_start)
        val_end = pd.Timestamp(val_end)

        train_end = val_start - pd.Timedelta(
            days=PREDICTION_WINDOW_DAYS
        )

        train_df = df[
            df[DATE_COLUMN] < train_end
        ].copy()

        val_df = df[
            (df[DATE_COLUMN] >= val_start)
            & (df[DATE_COLUMN] <= val_end)
        ].copy()

        folds.append({
            "train": train_df,
            "validation": val_df
        })

    return folds