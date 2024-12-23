import os
import re
from os.path import exists, join

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils import (
    compute_vibration,
    generate_spectrogram,
    load_audio,
    lowpass_filter,
    remove_principal_components,
)


class BaseDataset(Dataset):
    def __init__(self, data_folders, transform=None):
        """
        Args:
            data_folders: The folders containing the dataset
            transform: The transformations to apply to the images
        """
        self.data_folders = data_folders
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for folder in data_folders:
            print(f"Loading data from folder {folder}")
            inputs, labels = self.load_data_from_folder(folder)
            self.image_paths.extend(inputs)
            self.labels.append(labels)

        print("Processing labels...")
        self.process_labels()

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
        raise NotImplementedError

    def process_labels(self):
        self.labels = np.hstack(self.labels)
        self.normalize_labels()

    def normalize_labels(self):
        self.labels = (self.labels - self.labels.min()) / (
            self.labels.max() - self.labels.min()
        )


class AudioVisualDataset(BaseDataset):

    def __init__(
        self, data_folders, sample_freq, lowpass_freq=0, pca_drop=0, transform=None
    ):
        """
        Args:
            data_folders: The folders containing the dataset
            sample_freq: The sample frequency of the audio files
            lowpass_freq: The frequency to use for the low-pass (ignore if 0)
            pca_drop: The number of principal components to remove from the labels
            transform: The transformations to apply to the images
        """

        self.sample_freq = sample_freq
        self.lowpass_freq = lowpass_freq
        self.pca_drop = pca_drop
        super().__init__(data_folders, transform)

    def create_labels(self, folder_path, valid_ids):

        audio_file = join(folder_path, "audio.wav")
        trajectory_file = join(folder_path, "trajectory.csv")
        wav = load_audio(audio_file, self.sample_freq)
        if self.lowpass_freq > 0:
            wav = lowpass_filter(wav, self.lowpass_freq, self.sample_freq)
        trajectory = pd.read_csv(trajectory_file).iloc[valid_ids]

        spec_db = generate_spectrogram(
            wav, self.sample_freq, self.sample_freq // 4, self.sample_freq, db=True
        )
        spec_time = np.arange(0, spec_db.shape[1]) * 0.25
        timestamps = (trajectory["timestamp"] - trajectory["timestamp"].iloc[0]) / 1e9
        indices = np.array(
            [np.argmin(np.abs(spec_time - timestamp)) for timestamp in timestamps]
        )

        return spec_db[:, indices]

    def process_labels(self):

        self.labels = np.hstack(self.labels)
        if self.pca_drop > 0:
            self.labels = remove_principal_components(self.labels, self.pca_drop)
        self.normalize_labels()


class DownstreamDataset(BaseDataset):
    def __init__(self, data_folders, metric, transform=None):
        """
        Args:
            data_folders: The folders containing the dataset
            metric: The metric to use for the labels
            transform: The transformations to apply to the images
        """

        self.metric = metric
        super().__init__(data_folders, transform)

    def create_labels(self, folder_path, valid_ids):

        if self.metric == "vibration":
            imu_data = pd.read_csv(join(folder_path, "imu.csv"))
            metric_df = compute_vibration(imu_data)
        else:
            raise ValueError(f"Invalid metric: {self.metric}")

        trajectory_file = join(folder_path, "trajectory.csv")
        trajectory = pd.read_csv(trajectory_file).iloc[valid_ids]
        indices = np.array(
            [
                np.argmin(np.abs(metric_df["timestamp"] - timestamp))
                for timestamp in trajectory["timestamp"]
            ]
        )
        labels = metric_df["metric"][indices].values

        return torch.tensor(labels).unsqueeze(0)
