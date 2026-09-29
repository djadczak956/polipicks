import torch

from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from config import (
    FEATURE_COLUMNS,
    SECTOR_COLUMNS,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    THRESHOLD
)

from dataset import PoliPickDataset
from model import PoliPickModel
from splits import walk_forward_splits
from train import train_model
from evaluate import evaluate_model


def run_walk_forward(df):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    folds = walk_forward_splits(df)

    for fold_number, fold in enumerate(folds, start=1):

        print(f"\nFold {fold_number}")

        train_df = fold["train"].copy()
        val_df = fold["validation"].copy()

        scaler = StandardScaler()

        train_df[FEATURE_COLUMNS] = scaler.fit_transform(
            train_df[FEATURE_COLUMNS]
        )

        val_df[FEATURE_COLUMNS] = scaler.transform(
            val_df[FEATURE_COLUMNS]
        )

        train_dataset = PoliPickDataset(
            train_df,
            FEATURE_COLUMNS,
            SECTOR_COLUMNS
        )

        val_dataset = PoliPickDataset(
            val_df,
            FEATURE_COLUMNS,
            SECTOR_COLUMNS
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        model = PoliPickModel(
            input_size=len(FEATURE_COLUMNS),
            num_sectors=len(SECTOR_COLUMNS)
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
            device,
            THRESHOLD
        )

        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1: {f1:.4f}")


# Later:
#
# df = your_prepared_dataframe
# run_walk_forward(df)