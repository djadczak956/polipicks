import torch
from torch.utils.data import Dataset


class PoliPickDataset(Dataset):

    def __init__(self, dataframe, feature_columns, target_columns):

        self.features = torch.tensor(
            dataframe[feature_columns].values,
            dtype=torch.float32
        )

        self.targets = torch.tensor(
            dataframe[target_columns].values,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]