import os
import re
from os.path import exists, join

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, Subset
from utils import remove_principal_components, generate_spectrogram, load_audio


class AudioVisualDataModule(LightningDataModule):
    def __init__(
        self,
        input_size,
        data_folders,
        sample_freq, 
        pca_drop,
        batch_size,
        split_ratio=10,
        split_seed=42,
        k=1,
    ):
        """
        Args:
            input_size: The size of the input images
            data_folders: The folders containing the dataset
            sample_freq: The sample frequency of the audio files
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

        self.full_dataset = AudioVisualDataset(data_folders, sample_freq, pca_drop, transform)
        self.kfolds = KFold(n_splits=split_ratio, shuffle=True, random_state=split_seed)
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
        self.test_data = Subset(self.full_dataset, np.arange(len(self.full_dataset)))

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count() or 1)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count() or 1)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count() or 1)


class AudioVisualDataset(Dataset):

    def __init__(self, data_folders, sample_freq, pca_drop, transform=None):
        """
        Args:
            data_folders: The folders containing the dataset
            sample_freq: The sample frequency of the audio files
            pca_drop: The number of principal components to remove from the labels
            transform: The transformations to apply to the images
        """

        self.transform = transform
        self.sample_freq = sample_freq
        self.image_paths = []
        self.labels = []

        for folder in data_folders:
            print(f"Loading data from folder {folder}")
            inputs, labels = self.load_data_from_folder(folder)
            self.image_paths.extend(inputs)
            self.labels.append(labels)
        
        self.process_labels(pca_drop)


    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):

        image = cv2.imread(self.image_paths[idx])   
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        
        label = self.labels[:, idx]
        return image, label
    

    def load_data_from_folder(self, folder_path):
        inputs, ids = self.get_inputs(folder_path)
        labels = self.create_labels(folder_path, ids)
        return inputs, labels  


    def get_inputs(self, folder_path):
    
        patches_folder = join(folder_path, "patches")
        ids, input_files = [], []
        folder_ids = sorted(np.array(os.listdir(patches_folder), dtype=int))
        for folder_id in folder_ids:
            full_path = join(patches_folder, str(folder_id))
            if not exists(full_path) or len(os.listdir(full_path)) == 0:
                raise ValueError(f"Folder {full_path} is empty")
            closest_patch = self.get_closest_patch(full_path)
            if closest_patch:
                ids.append(folder_id)
                input_files.append(join(full_path, closest_patch))
        return input_files, ids     


    def get_closest_patch(self, folder_path):
    
        if not exists(folder_path) or len(os.listdir(folder_path)) == 0:
            return None
    
        min_dist = 10.0
        closest_patch = None
        pattern = re.compile(r"_\d+\.\d+")
        for file in os.listdir(folder_path):
            match = pattern.search(file)
            if file.endswith(".png") and match:
                distance = float(match.group()[1:])
                if distance < min_dist:
                    min_dist = distance
                    closest_patch = file

        return closest_patch 
        

    def create_labels(self, folder_path, valid_ids):
    
        audio_file = join(folder_path, "audio.wav")
        trajectory_file = join(folder_path, "trajectory.csv")
        wav = load_audio(audio_file, self.sample_freq)
        trajectory = pd.read_csv(trajectory_file).iloc[valid_ids]

        spec_db = generate_spectrogram(wav, self.sample_freq, self.sample_freq//4, self.sample_freq, db=True)
        spec_time = np.arange(0, spec_db.shape[1]) * 0.25
        timestamps = (trajectory["timestamp"] - trajectory["timestamp"].iloc[0]) / 1e9
        indices = np.array([np.argmin(np.abs(spec_time - timestamp)) for timestamp in timestamps])

        return spec_db[:, indices]
    

    def process_labels(self, pca_drop=0):

        self.labels = np.hstack(self.labels)
        if pca_drop > 0:
            self.labels = remove_principal_components(self.labels, pca_drop)

        # Display the labels as a spectrogram
        import librosa.display
        import matplotlib.pyplot as plt

        librosa.display.specshow(self.labels, sr=self.sample_freq, x_axis='s', y_axis='hz', cmap='inferno')
        plt.title("Labels Spectrogram")
        plt.show()
    
