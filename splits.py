import pandas as pd


def walk_forward_splits(df, date_column="prediction_date"):

    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])

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

        train_end = val_start - pd.Timedelta(days=21)

        train_df = df[
            df[date_column] < train_end
        ].copy()

        val_df = df[
            (df[date_column] >= val_start)
            & (df[date_column] <= val_end)
        ].copy()

        folds.append({
            "train": train_df,
            "validation": val_df
        })

    return folds