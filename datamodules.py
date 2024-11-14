import os
import re
from os.path import exists, join

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from lightning.pytorch import LightningDataModule
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, Subset


class TerrainPatchDataModule(LightningDataModule):
    def __init__(
        self,
        input_size,
        data_folder,
        batch_size,
        distance_bounds=None,
        split_ratio=10,
        split_seed=42,
        k=1,
    ):
        """
        Args:
            input_size: The size of the input images
            data_folder: The folder containing the dataset
            batch_size: The batch size to use for training
            distance_bounds: The min-max distance to consider for the patches (default=None)
            split_ratio: The number of parts to split the dataset into (default=10)
            split_seed: The seed to use for the random split (default=42)
            k: The k-th fold to use for training (default=1)
            blur: The size of the kernel for the Blur augmentation (default=1)
        """

        super().__init__()
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.split_seed = split_seed
        self.k = k
        self.distance_bounds = distance_bounds

        self.train_transform = A.Compose(
            [
                A.Resize(width=input_size, height=input_size, p=1),
                A.Normalize(normalization="min_max", p=1),
                ToTensorV2(),
            ]
        )

        self.test_transform = A.Compose(
            [
                A.Resize(width=input_size, height=input_size, p=1),
                A.Normalize(normalization="min_max", p=1),
                ToTensorV2(),
            ]
        )

        self.full_dataset = TerrainPatchDataset(self.data_folder, self.train_transform, self.distance_bounds)
        self.kfolds = KFold(n_splits=self.split_ratio, shuffle=True, random_state=self.split_seed)
        self.splits = [split for split in self.kfolds.split(self.full_dataset)]
        print(f"Dataset size: {len(self.full_dataset)}")

    def prepare_data(self):
        # Download data if needed
        pass

    def setup(self, stage: str):
        print("------------------------------------------------------")
        print(f"K: {self.k}")
        print("------------------------------------------------------")

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


class TerrainPatchDataset(Dataset):
    def __init__(self, data_folder, transform=None, distance_bounds=[0, 10]):
        self.patches_folder = os.path.join(data_folder, "patches")
        self.labels_folder = os.path.join(data_folder, "labels")
        self.transform = transform
        self.distance_bounds = distance_bounds
        self.ids = os.listdir(self.patches_folder)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        folder_id = self.ids[idx]
        folder_path = join(self.patches_folder, str(folder_id))

        # Pick a random valid image from the folder
        valid_files = self.get_valid_files(folder_path)
        random_idx = np.random.randint(len(valid_files))
        image_file = join(folder_path, valid_files[random_idx])
        image = cv2.imread(image_file)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        
        # TODO: Generate labels folder (spectrograms)
        label = None

        return image, label
    
    def get_valid_files(self, folder_path):
        if not exists(folder_path) or len(os.listdir(folder_path)) == 0:
            return []
        if self.distance_bounds is None:
            return os.listdir(folder_path)

        valid_files = []
        pattern = re.compile(r"_\d+\.\d+")
        for file in os.listdir(folder_path):
            match = pattern.search(file)
            if file.endswith(".png") and match:
                distance = float(match.group()[1:])
                if self.distance_bounds[0] <= distance <= self.distance_bounds[1]:
                    valid_files.append(file)

        return valid_files
