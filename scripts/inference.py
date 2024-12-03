import sys
from os.path import dirname, join, realpath
sys.path.append(join(dirname(__file__), ".."))

import torch
from datamodules import AudioVisualDataModule
import matplotlib.pyplot as plt
import numpy as np

############# PARAMETERS #############

SCRIPT_DIR = dirname(realpath(__file__))
MODEL_PATH = join(SCRIPT_DIR, "..", "output", "training", "2024-11-30_16-34-09", "model.pt")

DATA_FOLDERS = [
    join(SCRIPT_DIR, "..", "data", "sequence2"),
]
SAMPLE_FREQ = 4096
INPUT_SIZE = 256
BATCH_SIZE = 1024
DATA_SPLIT = 10  # Number of parts to split the dataset into
SEED = 42  # Seed for the random split

######################################

if __name__ == "__main__":

    # Check if CUDA is available
    torch.set_float32_matmul_precision("medium")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the test datamodules
    test_datamodule = AudioVisualDataModule(
        input_size=INPUT_SIZE,
        data_folders=DATA_FOLDERS,
        sample_freq=SAMPLE_FREQ,
        batch_size=BATCH_SIZE,
        split_ratio=10,
        split_seed=SEED,
    )
    test_datamodule.setup("test")

    # Load model with torchscript
    model = torch.jit.load(MODEL_PATH)
    model.to(device)
    model.eval()

    for batch in test_datamodule.test_dataloader():
        inputs, labels = batch
        inputs = inputs.to(device).float()
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)

        print(f"Inputs: {inputs.shape}")
        print(f"Labels: {labels.shape}")
        print(f"Outputs: {outputs.shape}")

        # Transpose the outputs and labels
        outputs_transposed = outputs.permute(1, 0).cpu().numpy()
        labels_transposed = labels.permute(1, 0).cpu().numpy()

        # Plot the first output and label
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(labels_transposed, cmap="inferno")   
        axes[0].set_title("Label")
        axes[1].imshow(outputs_transposed, cmap="inferno")
        axes[1].set_title("Output")
        plt.tight_layout()
        # plt.colorbar()
        plt.show()


