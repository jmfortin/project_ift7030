import os

import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

from datasets import AudioVisualDataset, DownstreamDataset


class AudioVisualDataModule(LightningDataModule):
    def __init__(
        self,
        input_size,
        batch_size,
        data_folders,
        sample_freq, 
        lowpass_freq=0,
        pca_drop=0,
        split_ratio=10,
        split_seed=42,
        k=1,
    ):
        """
        Args:
            input_size: The size of the input images
            data_folders: The folders containing the dataset
            sample_freq: The sample frequency of the audio files
            lowpass_freq: The frequency to use for the low-pass (ignore if 0)
            pca_drop: The number of principal components to remove from the labels
            batch_size: The batch size to use for training
            split_ratio: The number of parts to split the dataset into (default=10)
            split_seed: The seed to use for the random split (default=42)
            k: The k-th fold to use for training (default=1)
            blur: The size of the kernel for the Blur augmentation (default=1)
        """

        super().__init__()
        self.batch_size = batch_size
        self.k = k

        transform = A.Compose(
            [
                A.Resize(width=input_size, height=input_size, p=1),
                A.Normalize(normalization="min_max", p=1),
                ToTensorV2(),
            ]
        )

        self.full_dataset = AudioVisualDataset(data_folders, sample_freq, lowpass_freq, pca_drop, transform)
        self.kfolds = KFold(n_splits=split_ratio, shuffle=True, random_state=split_seed)
        self.splits = [split for split in self.kfolds.split(self.full_dataset)]
        print(f"Dataset size: {len(self.full_dataset)}")

    def prepare_data(self):
        # Download data if needed
        pass

    def setup(self, stage: str):
        train_indices, val_indices = self.splits[self.k]
        self.train_data = Subset(self.full_dataset, train_indices)
        self.val_data = Subset(self.full_dataset, val_indices)
        self.test_data = Subset(self.full_dataset, val_indices)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count() or 1)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count() or 1)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count() or 1)
    

class DownstreamDataModule(LightningDataModule):
    def __init__(
        self,
        input_size,
        batch_size,
        data_folders,
        metric,
        split_ratio=10,
        split_seed=42,
        k=1,
    ):
        """
        Args:
            input_size: The size of the input images
            batch_size: The batch size to use for training
            data_folders: The folders containing the dataset
            metric: The metric to use for the labels
            split_ratio: The number of parts to split the dataset into (default=10)
            split_seed: The seed to use for the random split (default=42)
            k: The k-th fold to use for training (default=1)
        """

        super().__init__()
        self.batch_size = batch_size
        self.k = k

        transform = A.Compose(
            [
                A.Resize(width=input_size, height=input_size, p=1),
                A.Normalize(normalization="min_max", p=1),
                ToTensorV2(),
            ]
        )

        self.full_dataset = DownstreamDataset(data_folders, metric, transform)
        self.kfolds = KFold(n_splits=split_ratio, shuffle=True, random_state=split_seed)
        self.splits = [split for split in self.kfolds.split(self.full_dataset)]
        print(f"Dataset size: {len(self.full_dataset)}")

    def prepare_data(self):
        # Download data if needed
        pass

    def setup(self, stage: str):
        train_indices, val_indices = self.splits[self.k]
        self.train_data = Subset(self.full_dataset, train_indices)
        self.val_data = Subset(self.full_dataset, val_indices)
        self.test_data = Subset(self.full_dataset, val_indices)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count() or 1)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count() or 1)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count() or 1)


