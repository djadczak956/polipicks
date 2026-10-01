import pandas as pd
import torch

from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from config.model_config import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS
)

from src.dataset import PoliPickDataset
from src.model import PoliPickModel
from src.splits import walk_forward_splits
from src.train import train_model
from src.evaluate import evaluate_model


def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def run_walk_forward(df):

    device = get_device()

    print(f"Using device: {device}")
    print(f"Total model rows: {len(df)}")

    folds = walk_forward_splits(df)

    print(f"Walk-forward folds: {len(folds)}")

    for fold_number, fold in enumerate(
        folds,
        start=1
    ):

        print(f"\n----------------------")
        print(f"Fold {fold_number}")
        print(f"----------------------")

        train_df = fold["train"].copy()
        val_df = fold["validation"].copy()

        print(f"Training rows: {len(train_df)}")
        print(f"Validation rows: {len(val_df)}")

        if train_df.empty or val_df.empty:
            print("Skipping fold because dataset is empty.")
            continue

        scaler = StandardScaler()

        train_df[FEATURE_COLUMNS] = (
            scaler.fit_transform(
                train_df[FEATURE_COLUMNS]
            )
        )

        val_df[FEATURE_COLUMNS] = (
            scaler.transform(
                val_df[FEATURE_COLUMNS]
            )
        )

        train_dataset = PoliPickDataset(
            train_df
        )

        val_dataset = PoliPickDataset(
            val_df
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
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
            EPOCHS
        )

        precision, recall, f1 = evaluate_model(
            model,
            val_loader,
            device
        )

        print()
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1:        {f1:.4f}")


def main():

    print("Loading model data...")

    df = pd.read_parquet(
        "data/processed/model_data.parquet"
    )

    run_walk_forward(df)


if __name__ == "__main__":
    main()