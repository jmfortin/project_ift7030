import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import ResNet18_Weights, resnet18, swin_v2_t

np.set_printoptions(precision=3)


class AudioVisualNet(LightningModule):

    def __init__(self, lr=0.0001):
        super(AudioVisualNet, self).__init__()

        self.lr = lr
        torch.manual_seed(42)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.forward(images.half())

        loss = F.mse_loss(outputs, targets.float(), reduction="mean")
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.forward(images.half())

        loss = F.mse_loss(outputs, targets.float(), reduction="mean")
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.forward(images.half())

        loss = F.mse_loss(outputs, targets.float(), reduction="mean")
        self.log("test_loss", loss, prog_bar=False)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)
        return [optimizer], [{"scheduler": scheduler, "monitor": "val_loss"}]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(metric)


class AudioVisualResNet(AudioVisualNet):

    def __init__(self, output_size=1, lr=0.0001):
        super(AudioVisualResNet, self).__init__(lr=lr)

        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Predict a spectrogram of the audio signal
        num_features = self.model.fc.in_features
        self.model.fc  = torch.nn.Sequential(
            torch.nn.Linear(num_features, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, output_size),
        )
        self.save_hyperparameters()

    def forward(self, x):
        x = self.model(x)
        return x


class AudioVisualSwin(AudioVisualNet):

    def __init__(self, output_size=1, lr=0.0001):
        super(AudioVisualSwin, self).__init__(lr=lr)

        self.model = swin_v2_t(weights="IMAGENET1K_V1")

        # Predict a spectrogram of the audio signal
        num_features = self.model.head.in_features
        self.model.head = torch.nn.Sequential(
            torch.nn.Linear(num_features, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, output_size)
        )

        self.save_hyperparameters()

    def forward(self, x):
        x = self.model(x)
        return x
