import torch
import torch.nn as nn

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


def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def train_one_epoch(model, dataloader, optimizer, device):

    model.train()

    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0

    for features, targets in dataloader:

        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward propagation
        logits = model(features)

        loss = criterion(logits, targets)

        # Back propagation
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def train_model(model, dataloader, optimizer, device, epochs):

    for epoch in range(epochs):

        loss = train_one_epoch(
            model,
            dataloader,
            optimizer,
            device
        )

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"Loss: {loss:.4f}"
        )


def fit_fold(train_df, device):

    train_df = train_df.copy()

    scaler = StandardScaler()

    train_df[FEATURE_COLUMNS] = (
        scaler.fit_transform(
            train_df[FEATURE_COLUMNS]
        )
    )

    train_loader = DataLoader(
        PoliPickDataset(train_df),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
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

    return model, scaler
