import sys
from os.path import dirname, join, realpath
sys.path.append(join(dirname(__file__), ".."))

import torch
from datamodules import AudioVisualDataModule
from networks import AudioVisualResNet, AudioVisualSwin
import yaml
from utils import display_spectograms, inference_on_datamodule
import os
from datetime import datetime

############# PARAMETERS #############

SCRIPT_DIR = dirname(realpath(__file__))
OUTPUT_DIR = join(SCRIPT_DIR, "..", "output", "training")

# Find the latest folder in the output/training directory
latest_dir = max(os.listdir(OUTPUT_DIR), key=lambda d: datetime.strptime(d, "%Y-%m-%d_%H-%M-%S"))
RESULTS_PATH = join(OUTPUT_DIR, latest_dir)
# RESULTS_PATH = join(OUTPUT_DIR, "2024-12-03_13-29-00")
CHECKPOINT_PATH = join(RESULTS_PATH, "model.pt")

######################################

if __name__ == "__main__":

    # Load details.yaml
    with open(join(RESULTS_PATH, "details.yaml"), "r") as file:
        details = yaml.safe_load(file)

    # Generate model.pt from model.ckpt if it does not exist
    if not os.path.exists(CHECKPOINT_PATH):
        if details["model_type"] == "ResNet":
            model = AudioVisualResNet(output_size=details["output_size"], lr=details["learning_rate"])
        elif details["model_type"] == "Swin":
            model = AudioVisualSwin(output_size=details["output_size"], lr=details["learning_rate"])
        else:
            raise ValueError("Invalid model type.")
        
        best_checkpoint = torch.load(join(RESULTS_PATH, "model.ckpt"))
        model.load_state_dict(best_checkpoint["state_dict"])
        model.to_torchscript(join(RESULTS_PATH, "model.pt"))
      
    # Check if CUDA is available
    torch.set_float32_matmul_precision("medium")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the test datamodules
    test_datamodule = AudioVisualDataModule(
        input_size=details["input_size"],
        data_folders=details["data_folders"],
        sample_freq=details["sample_freq"],
        pca_drop=details["pca_drop"],
        batch_size=details["batch_size"],
        split_ratio=details["data_split"]  ,
        split_seed=details["seed"],
    )
    test_datamodule.setup("test")

    # Load model with torchscript
    model = torch.jit.load(CHECKPOINT_PATH)
    model.to(device)
    model.eval()

    print("Starting inference...")
    labels, outputs = inference_on_datamodule(model, test_datamodule, device)

    # Plot the output and label
    display_spectograms(
        specs=[labels, outputs], sample_freq=details["sample_freq"], 
        titles=["Labels", "Predictions"], 
        save_path=join(RESULTS_PATH, "predictions.png")
    )


