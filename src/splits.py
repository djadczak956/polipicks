import pandas as pd

from config.model_config import (
    PREDICTION_WINDOW_DAYS,
    DATE_COLUMN,
    TUNING_DAYS
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


def split_for_tuning(train_df):

    # Hold out the last TUNING_DAYS of the training period to pick the
    # threshold. Rows just before the cutoff are dropped because their
    # 21-day target windows would overlap the tuning period.
    tune_start = (
        train_df[DATE_COLUMN].max()
        - pd.Timedelta(days=TUNING_DAYS)
    )

    fit_df = train_df[
        train_df[DATE_COLUMN]
        < tune_start - pd.Timedelta(days=PREDICTION_WINDOW_DAYS)
    ]

    tune_df = train_df[
        train_df[DATE_COLUMN] >= tune_start
    ]

    return fit_df, tune_df
