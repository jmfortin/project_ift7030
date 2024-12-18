import argparse
import os
from datetime import datetime
from os.path import dirname, join, realpath, relpath

import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import WandbLogger

from datamodules import DownstreamDataModule
from networks import AudioVisualResNet, AudioVisualSwin
import yaml

from utils import display_spectograms, inference_on_datamodule
import matplotlib.pyplot as plt

############# PARAMETERS #############

SCRIPT_DIR = dirname(realpath(__file__))
SAVE_PATH = join(SCRIPT_DIR, "..", "output", "finetune")
CHECKPOINT_PATH = join(SCRIPT_DIR, "..", "output", "training", "2024-12-18_11-59-46")
# CHECKPOINT_PATH = None 
FREEZE_BACKBONE = False
METRIC = "vibration"

PROJECT_NAME = "ift7030"
NUM_FOLDS = 5  # Number of folds for cross-validation

parser = argparse.ArgumentParser(description="Train model for audio prediction from terrain patches.")
parser.add_argument("--folds", type=int, help="Number of folds for cross-validation.", default=NUM_FOLDS)
parser.add_argument("--project", type=str, help="WandB project name.", default=PROJECT_NAME)
args = parser.parse_args()

DATA_FOLDERS = [
    join(SCRIPT_DIR, "..", "data", "sequence1"),
    join(SCRIPT_DIR, "..", "data", "sequence2"),
    join(SCRIPT_DIR, "..", "data", "sequence3"),
    join(SCRIPT_DIR, "..", "data", "sequence4"),
    join(SCRIPT_DIR, "..", "data", "sequence5"),
    join(SCRIPT_DIR, "..", "data", "sequence6"),
]
MODEL_TYPE = "ResNet"  # Choose between "ResNet" and "Swin"
INPUT_SIZE = 256
OUTPUT_SIZE = 1
BATCH_SIZE = 128
NUM_EPOCHS = 100
PATIENCE = 10  # Set to NUM_EPOCHS to disable early stopping
LEARNING_RATE = 0.001

DATA_SPLIT = 10  # Number of parts to split the dataset into
SEED = 42  # Seed for the random split

######################################

if __name__ == "__main__":

    # Check if CUDA is available
    torch.set_float32_matmul_precision("medium")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Weights & Biases
    os.environ["WANDB_SILENT"] = "true"
    tags = [MODEL_TYPE, "finetune"]

    # Load the dataset
    datamodule = DownstreamDataModule(
        input_size=INPUT_SIZE,
        data_folders=DATA_FOLDERS,
        metric=METRIC,
        batch_size=BATCH_SIZE,
        split_ratio=DATA_SPLIT,
        split_seed=SEED,
    )

    # Create a save folder matching the date and time
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_folder = join(SAVE_PATH, time_str)
    os.makedirs(save_folder, exist_ok=True)

    # Save details to a YAML file
    details = {
        "checkpoint_path": CHECKPOINT_PATH,
        "freeze": FREEZE_BACKBONE,
        "metric": METRIC,
        "data_folders": DATA_FOLDERS,
        "input_size": INPUT_SIZE,
        "output_size": OUTPUT_SIZE,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "model_type": MODEL_TYPE,
        "data_split": DATA_SPLIT,
        "seed": SEED,
    }

    with open(join(save_folder, "details.yaml"), "w") as file:
        yaml.dump(details, file)

    # Start cross-validation
    for k in range(args.folds):
        if args.folds > 1:
            print("\n--------------------------------")
            print(f"  Fold {k + 1} / {args.folds}")
            print("--------------------------------\n")
            fold_save_folder = join(save_folder, f"fold_{k+1}")
        else:
            fold_save_folder = save_folder

        # Load the checkpoint (if provided)
        if CHECKPOINT_PATH is not None:
            ckpt_path = join(CHECKPOINT_PATH, f"fold_{k+1}", "model.ckpt")
            print(f"Loading checkpoint from {relpath(ckpt_path, SCRIPT_DIR)}")
            ckpt_details = yaml.safe_load(open(join(CHECKPOINT_PATH, "details.yaml"), "r"))
            if ckpt_details["model_type"] == "ResNet":
                model = AudioVisualResNet.load_from_checkpoint(ckpt_path)
            elif ckpt_details["model_type"] == "Swin":
                model = AudioVisualSwin.load_from_checkpoint(ckpt_path)
            else:
                raise ValueError("Invalid model type.")
        else:
            print("No checkpoint provided. Creating a new model.")
            if MODEL_TYPE == "ResNet":
                model = AudioVisualResNet(output_size=OUTPUT_SIZE, lr=LEARNING_RATE)
            elif MODEL_TYPE == "Swin":
                model = AudioVisualSwin(output_size=OUTPUT_SIZE, lr=LEARNING_RATE)
            else:
                raise ValueError("Invalid model type.")

        # Freeze the model and replace the head
        model.replace_head(OUTPUT_SIZE)
        if FREEZE_BACKBONE:
            model.freeze_backbone()

        # Define logger
        logger = WandbLogger(
            log_model=False,
            project=args.project,
            save_dir=fold_save_folder,
            name=f"{time_str}_fold_{k+1}",
            tags=tags,
        )

        # Define callbacks
        callbacks = [
            EarlyStopping(monitor="val_loss", mode="min", patience=PATIENCE),
            ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, dirpath=fold_save_folder, filename="model"),
            LearningRateMonitor(logging_interval="step"),
            RichProgressBar(leave=True),
        ]

        trainer = Trainer(
            max_epochs=NUM_EPOCHS,
            accelerator="gpu",
            logger=logger,
            log_every_n_steps=1,
            enable_checkpointing=True,
            callbacks=callbacks,
            precision="16-mixed",
        )

        # Train the model
        datamodule.k = k
        trainer.fit(model, datamodule)

        # Evaluate the model
        # metrics = trainer.test(model, datamodule)
        labels, outputs = inference_on_datamodule(model, datamodule, device)

        # Plot the output and label and save the plots
        fig = plt.figure(figsize=(12, 6))
        plt.plot(labels[0, :], label="Labels")
        plt.plot(outputs[0, :], label="Predictions")
        plt.legend()
        plt.tight_layout()
        plt.savefig(join(fold_save_folder, "predictions.png"))

        # Save the model as a TorchScript file
        best_checkpoint = torch.load(join(fold_save_folder, "model.ckpt"), weights_only=True)
        model.load_state_dict(best_checkpoint["state_dict"])
        model.to_torchscript(join(fold_save_folder, "model.pt"))

        wandb.finish()
