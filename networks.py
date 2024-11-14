import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import ResNet18_Weights, resnet18, swin_v2_t

np.set_printoptions(precision=3)


class TerrainPatchNet(LightningModule):

    def __init__(self, lr=0.0001):
        super(TerrainPatchNet, self).__init__()

        self.lr = lr
        torch.manual_seed(42)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.forward(images)

        loss = F.mse_loss(outputs, targets.float())
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.forward(images)

        loss = F.mse_loss(outputs, targets.float(), reduction="none")
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.forward(images)

        loss = F.mse_loss(outputs, targets.float(), reduction="none")
        self.log("test_loss", loss, prog_bar=False)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)
        return [optimizer], [{"scheduler": scheduler, "monitor": "val_loss"}]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(metric)


class TerrainPatchResNet(TerrainPatchNet):

    def __init__(self, image_size=64, output_size=1, lr=0.0001):
        super(TerrainPatchResNet, self).__init__(lr=lr)

        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # TODO - Add the heads for the model
        # It should predict a spectrogram of the audio signal

        self.save_hyperparameters()

    def forward(self, x):
        x = self.model(x)
        return torch.cat([head(x) for head in self.heads], dim=1)


class TerrainPatchSwin(TerrainPatchNet):

    def __init__(self, image_size=64, output_size=1, lr=0.0001):
        super(TerrainPatchSwin, self).__init__(lr=lr)

        self.model = swin_v2_t(weights="IMAGENET1K_V1")

        # TODO - Add the heads for the model
        # It should predict a spectrogram of the audio signal

        self.save_hyperparameters()

    def forward(self, x):
        x = self.model(x)
        return torch.cat([head(x) for head in self.heads], dim=1)
