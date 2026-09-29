import torch
from torch.utils.data import Dataset

from config.model_config import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS
)


class PoliPickDataset(Dataset):

    def __init__(self, dataframe):

        self.features = torch.tensor(
            dataframe[FEATURE_COLUMNS].values,
            dtype=torch.float32
        )

        self.targets = torch.tensor(
            dataframe[TARGET_COLUMNS].values,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]