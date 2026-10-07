import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, width, dropout):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
        )
        self.activation = nn.ReLU()

    def forward(self, x):

        return self.activation(x + self.layers(x))


class PoliPickModel(nn.Module):

    def __init__(self, input_size, num_sectors):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.25)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.20)

        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()

        self.output = nn.Linear(32, num_sectors)

    def forward(self, x):

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.relu3(x)

        logits = self.output(x)

        return logits


class ResidualPoliPickModel(nn.Module):

    def __init__(self, input_size, num_sectors):
        super().__init__()

        self.input = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.20),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(128, 0.20),
            ResidualBlock(128, 0.20),
        )
        self.output = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, num_sectors),
        )

    def forward(self, x):

        x = self.input(x)
        x = self.blocks(x)

        return self.output(x)