import argparse
import os
from datetime import datetime
from os.path import dirname, join, realpath

import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import WandbLogger

from datamodules import AudioVisualDataModule
from networks import AudioVisualResNet, AudioVisualSwin
import yaml

############# PARAMETERS #############

SCRIPT_DIR = dirname(realpath(__file__))
SAVE_PATH = join(SCRIPT_DIR, "..", "output", "training")

PROJECT_NAME = "ift7030"
NUM_FOLDS = 1  # Number of folds for cross-validation

parser = argparse.ArgumentParser(description="Train model for audio prediction from terrain patches.")
parser.add_argument("--folds", type=int, help="Number of folds for cross-validation.", default=NUM_FOLDS)
parser.add_argument("--mindist", type=float, help="Minimum distance from the robot.", default=0)
parser.add_argument("--maxdist", type=float, help="Maximum distance from the robot.", default=3)
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
SAMPLE_FREQ = 4096
INPUT_SIZE = 256
OUTPUT_SIZE = SAMPLE_FREQ//2 + 1
BATCH_SIZE = 128
NUM_EPOCHS = 100
PATIENCE = 5  # Set to NUM_EPOCHS to disable early stopping
LEARNING_RATE = 0.001
PCA_DROP = 0

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
    datamodule = AudioVisualDataModule(
        input_size=INPUT_SIZE,
        data_folders=DATA_FOLDERS,
        sample_freq=SAMPLE_FREQ,
        pca_drop=PCA_DROP,
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
        "data_folders": DATA_FOLDERS,
        "sample_freq": SAMPLE_FREQ,
        "input_size": INPUT_SIZE,
        "output_size": OUTPUT_SIZE,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "pca_drop": PCA_DROP,
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

        # Create an instance of the selected model
        if MODEL_TYPE == "ResNet":
            model = AudioVisualResNet(output_size=OUTPUT_SIZE, lr=LEARNING_RATE)
        elif MODEL_TYPE == "Swin":
            model = AudioVisualSwin(output_size=OUTPUT_SIZE, lr=LEARNING_RATE)
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
