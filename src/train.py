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
    EPOCHS,
    POS_WEIGHT,
    SEED
)

from src.dataset import PoliPickDataset
from src.model import PoliPickModel


def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def compute_pos_weight(targets, mode):

    # Rare sectors (Utilities ~2% positive) barely move an unweighted loss,
    # so the model learns to never predict them. pos_weight scales up the
    # loss on positive labels per sector.
    #   "none": no weighting
    #   "sqrt": sqrt(negatives / positives) -- best in the sweep
    #   "full": negatives / positives
    if mode == "none":
        return None

    positive_rate = (
        torch.as_tensor(targets, dtype=torch.float32)
        .mean(dim=0)
        .clamp(min=1e-3)
    )

    ratio = (1 - positive_rate) / positive_rate

    if mode == "sqrt":
        return torch.sqrt(ratio)

    if mode == "full":
        return ratio

    raise ValueError(f"Unknown POS_WEIGHT mode: {mode}")


def train_one_epoch(model, dataloader, optimizer, criterion, device):

    model.train()

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


def train_model(
    model,
    dataloader,
    optimizer,
    device,
    epochs,
    pos_weight=None,
    verbose=True
):

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=(
            pos_weight.to(device)
            if pos_weight is not None
            else None
        )
    )

    for epoch in range(epochs):

        loss = train_one_epoch(
            model,
            dataloader,
            optimizer,
            criterion,
            device
        )

        if verbose:
            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"Loss: {loss:.4f}"
            )


def fit_fold(train_df, device):

    torch.manual_seed(SEED)

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
        EPOCHS,
        pos_weight=compute_pos_weight(
            train_df[TARGET_COLUMNS].values,
            POS_WEIGHT
        ),
        verbose=False
    )

    return model, scaler
