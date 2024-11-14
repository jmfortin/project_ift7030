import argparse
import os
import sys
from datetime import datetime
from os.path import dirname, join, realpath

sys.path.append(join(dirname(__file__), ".."))

import torch
import wandb
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import WandbLogger

from neural_network.datamodules import TerrainPatchDataModule
from neural_network.networks_patch import TerrainPatchResNet, TerrainPatchSwin

############# PARAMETERS #############

SCRIPT_DIR = dirname(realpath(__file__))
SAVE_PATH = join(SCRIPT_DIR, "..", "output", "training")

PROJECT_NAME = "ift7030"
NUM_FOLDS = 1  # Number of folds for cross-validation

parser = argparse.ArgumentParser(description="Train model for audio prediction from terrain patches.")
parser.add_argument("--folds", type=int, help="Number of folds for cross-validation.", default=NUM_FOLDS)
parser.add_argument("--mindist", type=float, help="Minimum distance from the robot.", default=0)
parser.add_argument("--maxdist", type=float, help="Maximum distance from the robot.", default=5)
parser.add_argument("--project", type=str, help="WandB project name.", default=PROJECT_NAME)
args = parser.parse_args()

DATA_FOLDER = join(SCRIPT_DIR, "..", "data")
INPUT_SIZE = 256
BATCH_SIZE = 256
NUM_EPOCHS = 100
PATIENCE = 100  # Set to NUM_EPOCHS to disable early stopping
LEARNING_RATE = 0.00005

MODEL_TYPE = "ResNet"  # Choose between "CNN", "ResNet" and "Swin"
DATA_SPLIT = 10  # Number of parts to split the dataset into
SEED = 42  # Seed for the random split

######################################

if __name__ == "__main__":

    # Initialize Weights & Biases
    torch.set_float32_matmul_precision("medium")
    os.environ["WANDB_SILENT"] = "true"
    wandb.require("core")

    tags = [MODEL_TYPE]

    # Load the dataset
    datamodule = TerrainPatchDataModule(
        input_size=INPUT_SIZE,
        data_folder=DATA_FOLDER,
        batch_size=BATCH_SIZE,
        distance_bounds=[args.mindist, args.maxdist],
        split_ratio=DATA_SPLIT,
        split_seed=SEED,
    )

    # Create a save folder matching the date and time
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_folder = join(SAVE_PATH, time_str)
    os.makedirs(save_folder, exist_ok=True)

    # Start cross-validation
    for k in range(args.folds):
        if args.folds > 1:
            print("\n--------------------------------")
            print(f"  Fold {k + 1} / {args.folds}")
            print("--------------------------------\n")
            fold_save_folder = join(save_folder, f"fold_{k+1}")
        else:
            fold_save_folder = save_folder

        # Create an instance of the selected model
        if MODEL_TYPE == "ResNet":
            model = TerrainPatchResNet(image_size=INPUT_SIZE, lr=LEARNING_RATE)
        elif MODEL_TYPE == "Swin":
            model = TerrainPatchSwin(image_size=INPUT_SIZE, lr=LEARNING_RATE)
        else:
            raise ValueError("Invalid model type.")

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
        metrics = trainer.test(model, datamodule)

        # Save the model as a TorchScript file
        best_checkpoint = torch.load(join(fold_save_folder, "model.ckpt"))
        model.load_state_dict(best_checkpoint["state_dict"])
        model.to_torchscript(join(fold_save_folder, "model.pt"))

        wandb.finish()
