import torch
import torch.nn as nn


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
