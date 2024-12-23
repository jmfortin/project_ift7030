import sys
import os
from os.path import dirname, join, realpath
sys.path.append(join(dirname(__file__), ".."))

import torch
import yaml
from datetime import datetime
import numpy as np
from tqdm import tqdm
from datamodules import DownstreamDataModule, AudioVisualDataModule

############# PARAMETERS #############

SCRIPT_DIR = dirname(realpath(__file__))
OUTPUT_DIR = join(SCRIPT_DIR, "..", "output", "training")

# Find the latest folder in the output/training directory
latest_dir = max(os.listdir(OUTPUT_DIR), key=lambda d: datetime.strptime(d, "%Y-%m-%d_%H-%M-%S"))
RESULTS_PATH = join(OUTPUT_DIR, latest_dir)
RESULTS_PATH = join(SCRIPT_DIR, "..", "output", "finetune", "2024-12-20_12-02-19")

######################################

def load_test_datamodule(checkpoint_path):

    with open(join(checkpoint_path, "details.yaml"), "r") as file:
        details = yaml.safe_load(file)

    if "checkpoint_path" in details:
        test_datamodule = DownstreamDataModule(
            input_size=details["input_size"],
            data_folders=details["data_folders"],
            metric=details["metric"],
            batch_size=details["batch_size"],
            split_ratio=details["data_split"],
            split_seed=details["seed"],
        )
    else:
        test_datamodule = AudioVisualDataModule(
            input_size=details["input_size"],
            data_folders=details["data_folders"],
            sample_freq=details["sample_freq"],
            pca_drop=details["pca_drop"],
            batch_size=details["batch_size"],
            split_ratio=details["data_split"]  ,
            split_seed=details["seed"],
        )
    return test_datamodule


def inference_on_datamodule(model, datamodule, device):

    model.eval()
    model.to(device)
    datamodule.setup("test")
    
    all_labels = np.array([])
    all_outputs = np.array([])
    for batch in datamodule.test_dataloader():
        inputs, labels = batch
        inputs = inputs.to(device).float()
        labels = labels.to(device).float()

        with torch.no_grad():
            outputs = model(inputs)

        # Transpose the outputs and labels
        outputs = outputs.permute(1, 0).cpu().numpy()
        labels = labels.permute(1, 0).cpu().numpy()
        if all_labels.size == 0:
            all_labels = labels
            all_outputs = outputs
        else:
            all_labels = np.hstack((all_labels, labels))
            all_outputs = np.hstack((all_outputs, outputs))

    return all_labels, all_outputs


def run_inference_from_checkpoint(path):
      
    # Check if CUDA is available
    torch.set_float32_matmul_precision("medium")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the test datamodules
    test_datamodule = load_test_datamodule(path)

    all_labels = []
    all_outputs = []

    nb_folds = len([d for d in os.listdir(path) if d.startswith("fold_")])
    nb_folds = 2
    print(f"Starting inference for {nb_folds} folds")
    for k in tqdm(range(nb_folds)):
        if nb_folds > 1:
            fold_save_folder = join(path, f"fold_{k+1}")
        else:
            fold_save_folder = path
        fold_save_folder = join(path, f"fold_{k+1}")

        # Load model with torchscript
        model = torch.jit.load(join(fold_save_folder, "model.pt"))
        model.to(device)
        model.eval()

        test_datamodule.k = k
        labels, outputs = inference_on_datamodule(model, test_datamodule, device)
        all_labels.append(labels)
        all_outputs.append(outputs)

    return np.concatenate(all_labels, axis=1), np.concatenate(all_outputs, axis=1)


def main():
    labels, outputs = run_inference_from_checkpoint(RESULTS_PATH)
    rmse = np.sqrt(np.mean((labels - outputs) ** 2))
    print(f"Mean RMSE: {np.mean(rmse)}")
    print(f"Std RMSE: {np.std(rmse)}")
    print(f"Max RMSE: {np.max(rmse)}")
    print(f"Min RMSE: {np.min(rmse)}")
    print(f"Variance on predictions: {np.var(outputs)}")
    mean_variance_per_row = np.mean(np.var(outputs, axis=1))
    print(f"Mean Variance per Row: {mean_variance_per_row}")
    print("Inference done!")


if __name__ == "__main__":
    main()
