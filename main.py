import pandas as pd
import torch

from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from config.model_config import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    DATE_COLUMN,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    POS_WEIGHT,
    SEED,
    SECTORS
)

from src.dataset import PoliPickDataset
from src.model import PoliPickModel
from src.splits import walk_forward_splits, split_for_tuning
from src.train import train_model, compute_pos_weight, get_device
from src.evaluate import (
    predict_probabilities,
    choose_threshold,
    score_predictions
)


# Every fold's test-period probabilities, read by the dashboard's
# "Model performance" tab.
PREDICTIONS_OUT = "data/processed/model_predictions.parquet"


def fit_and_predict(train_df, eval_df, device):

    # Same seed for every fit so runs are repeatable.
    torch.manual_seed(SEED)

    train_df = train_df.copy()
    eval_df = eval_df.copy()

    scaler = StandardScaler()

    train_df[FEATURE_COLUMNS] = scaler.fit_transform(
        train_df[FEATURE_COLUMNS]
    )

    eval_df[FEATURE_COLUMNS] = scaler.transform(
        eval_df[FEATURE_COLUMNS]
    )

    train_loader = DataLoader(
        PoliPickDataset(train_df),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    eval_loader = DataLoader(
        PoliPickDataset(eval_df),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = PoliPickModel(
        input_size=len(FEATURE_COLUMNS),
        num_sectors=len(TARGET_COLUMNS)
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    train_model(
        model,
        train_loader,
        optimizer,
        device,
        EPOCHS,
        pos_weight=compute_pos_weight(
            train_df[TARGET_COLUMNS].values,
            POS_WEIGHT
        ),
        verbose=False
    )

    return predict_probabilities(
        model,
        eval_loader,
        device
    )


def run_walk_forward(df):

    device = get_device()

    print(f"Using device: {device}")
    print(f"Total model rows: {len(df)}")

    folds = walk_forward_splits(df)

    print(f"Walk-forward folds: {len(folds)}")

    results = []
    predictions = []

    for fold_number, fold in enumerate(
        folds,
        start=1
    ):

        train_df = fold["train"]
        test_df = fold["validation"]

        if train_df.empty or test_df.empty:
            print(f"\nFold {fold_number}: skipping, empty split")
            continue

        print(
            f"\nFold {fold_number}: "
            f"train {len(train_df)} rows, "
            f"test {len(test_df)} rows "
            f"(from {test_df[DATE_COLUMN].min().date()})"
        )

        # 1. Pick the threshold on the last year of training data.
        fit_df, tune_df = split_for_tuning(train_df)

        y_tune, p_tune = fit_and_predict(
            fit_df,
            tune_df,
            device
        )

        threshold = choose_threshold(
            y_tune,
            p_tune
        )

        # 2. Retrain on the whole training period, score the test period once.
        y_test, p_test = fit_and_predict(
            train_df,
            test_df,
            device
        )

        scores = score_predictions(
            y_test,
            (p_test >= threshold).astype(int)
        )

        print(
            f"  threshold={threshold:.2f}  "
            + "  ".join(
                f"{name}={value:.4f}"
                for name, value in scores.items()
            )
        )

        predictions.append(
            test_df[["memberId", DATE_COLUMN]]
            .reset_index(drop=True)
            .assign(fold=fold_number, threshold=threshold)
            .join(pd.DataFrame(
                y_test.astype(int),
                columns=[f"target_{s}" for s in SECTORS]
            ))
            .join(pd.DataFrame(
                p_test,
                columns=[f"prob_{s}" for s in SECTORS]
            ))
        )

        results.append({
            "fold": fold_number,
            "test_start": test_df[DATE_COLUMN].min().date(),
            "threshold": threshold,
            **scores,
        })

    summary = pd.DataFrame(results).set_index("fold")

    pd.concat(predictions, ignore_index=True).to_parquet(
        PREDICTIONS_OUT,
        index=False
    )

    print("\n======================")
    print("Summary")
    print("======================")
    print(
        f"lr={LEARNING_RATE} epochs={EPOCHS} "
        f"batch={BATCH_SIZE} weight_decay={WEIGHT_DECAY} "
        f"pos_weight={POS_WEIGHT} seed={SEED}\n"
    )
    print(summary.round(4).to_string())
    print(
        f"\nmean      precision={summary.precision.mean():.4f} "
        f"recall={summary.recall.mean():.4f} "
        f"f1={summary.f1.mean():.4f} "
        f"macro_f1={summary.macro_f1.mean():.4f}"
    )
    print(f"\nwrote {PREDICTIONS_OUT}")

    return summary


def main():

    print("Loading model data...")

    df = pd.read_parquet(
        "data/processed/model_data.parquet"
    )

    df[DATE_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN]
    )

    run_walk_forward(df)


if __name__ == "__main__":
    main()
