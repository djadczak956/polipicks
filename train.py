import torch.nn as nn


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