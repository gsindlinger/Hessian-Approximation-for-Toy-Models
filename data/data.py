from abc import ABC, abstractmethod
from os import name
from sklearn.discriminant_analysis import StandardScaler
from sklearn.discriminant_analysis import StandardScaler
from typing_extensions import override
from sklearn.datasets import make_regression, make_classification
import torch
from torch.utils.data import Dataset
from config.config import (
    DatasetConfig,
    RandomClassificationConfig,
    RandomRegressionConfig,
    UCIDatasetConfig,
)


class AbstractDataset(Dataset, ABC):
    def __init__(self, train_test_split=0.8, transform=None):
        super().__init__()
        self.train_test_split = train_test_split  # Default split, can be overridden
        self.transform = transform  # Placeholder for any transformations

    def split_dataset(self):
        """Split dataset into train and test sets. Assumes data to be normalized."""
        split_idx = int(len(self) * self.train_test_split)
        self.train_dataset = torch.utils.data.Subset(self, range(0, split_idx))
        self.test_dataset = torch.utils.data.Subset(self, range(split_idx, len(self)))
        return self.train_dataset, self.test_dataset

    def test_train_to_dataloader(self, batch_size=32, shuffle=True):
        """Split dataset into train and test sets and return dataloaders. Assumes data to be normalized."""
        if not hasattr(self, "train_dataset") or not hasattr(self, "test_dataset"):
            self.split_dataset()

        train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=shuffle
        )
        if self.train_test_split < 1.0:
            test_loader = torch.utils.data.DataLoader(
                self.test_dataset, batch_size=batch_size, shuffle=shuffle
            )
        else:
            test_loader = None
        return train_loader, test_loader

    @abstractmethod
    def input_dim(self) -> int:
        pass

    @abstractmethod
    def output_dim(self) -> int:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        pass


def create_dataset(config: DatasetConfig) -> AbstractDataset:
    """Create dataset from config."""
    dataset_map = {
        RandomRegressionConfig: RandomRegressionDataset,
        RandomClassificationConfig: RandomClassificationDataset,
        UCIDatasetConfig: UCIDataset,
    }

    dataset_cls = dataset_map.get(type(config))
    if dataset_cls is None:
        raise ValueError(f"Unknown dataset: {type(config).__name__}")

    dataset_kwargs = vars(config).copy()
    return dataset_cls(**dataset_kwargs)


class RandomRegressionDataset(AbstractDataset):
    def __init__(
        self, n_samples, n_features, n_targets, noise, random_state, train_test_split
    ):
        super().__init__(train_test_split=train_test_split)
        self.data, self.targets = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_targets=n_targets,
            noise=noise,
            random_state=random_state,
        )[:2]
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32).view(-1, 1)

    @override
    def __len__(self):
        return len(self.data)

    @override
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    @override
    def input_dim(self):
        return self.data.shape[1]

    @override
    def output_dim(self):
        return self.targets.shape[1]


class RandomClassificationDataset(AbstractDataset):
    def __init__(
        self,
        n_samples,
        n_features,
        n_informative,
        n_classes,
        random_state,
        train_test_split,
    ):
        super().__init__(train_test_split=train_test_split)
        self.data, self.targets = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_classes=n_classes,
            random_state=random_state,
        )[:2]
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.long)

    @override
    def __len__(self):
        return len(self.data)

    @override
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    @override
    def input_dim(self):
        return self.data.shape[1]

    @override
    def output_dim(self) -> int:
        return int(self.targets.max().item() + 1)


class UCIDataset(AbstractDataset):
    def __init__(self, name: str, train_test_split: float = 0.8):
        super().__init__(train_test_split=train_test_split)
        match name:
            case "energy":
                id = 242
            case _:
                raise ValueError(f"Unknown UCI dataset: {name}")

        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=id)
        X = dataset.data.features  # type: ignore  8 features, i.e. shape of X is pandas.DataFrame (768,8)
        Y = dataset.data.targets  # type: ignore  2 targets, i.e. shape of Y is pandas.DataFrame (768,2)

        scaler_X = StandardScaler()
        X = scaler_X.fit_transform(X)
        scaler_Y = StandardScaler()
        Y = scaler_Y.fit_transform(Y)

        self.data = X
        self.targets = Y

        self.train_test_split = train_test_split

        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32)

    @override
    def __len__(self):
        return len(self.data)

    @override
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    @override
    def input_dim(self):
        return self.data.shape[1]

    @override
    def output_dim(self):
        return self.targets.shape[1] if len(self.targets.shape) > 1 else 1
