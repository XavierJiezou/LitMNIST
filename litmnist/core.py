import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from torchmetrics.functional import accuracy


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = os.getcwd()):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.dims = (1, 28, 28)
        self.num_classes = 10
        self.batch_size = batch_size
        self.num_workders = num_workders

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=not os.path.exists('MNIST'))
        MNIST(self.data_dir, train=False, download=not os.path.exists('MNIST'))

    def setup(self, stage: str = None):
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(
                root=self.data_dir,
                train=True,
                transform=self.transform
            )
            self.mnist_train, self.mnist_val = random_split(
                dataset=mnist_full,
                lengths=[55000, 5000]
            )
        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(
                root=self.data_dir,
                train=False,
                transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.mnist_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workders,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.mnist_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.mnist_test,
            batch_size=self.batch_size,
            num_workers=self.num_workersk,
            persistent_workers=True
        )


class PlMnist(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.data_dir = os.getcwd()
        self.num_classes = 10
        self.channels = channels
        self.width = width
        self.height = height
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self.model(x), y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        pred = self.model(x)
        loss = F.cross_entropy(pred, y)
        acc = accuracy(pred.argmax(1), y)
        self.log('val_loss', loss)
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def make_dataloaders(batch_size: int = 32, num_workers: int = 16) -> dict:
    """Make a data loader for dictionary types.

    Args:
        batch_size (int, optional): How many samples per batch to load. Defaults to 32.
        num_workers (int, optional): How many subprocesses to use for data loading. Defaults to 16.

    Returns:
        dict: Keys of the dictionnary are taken from [train, val, test]. Corresponding value is a data loader.
    """


def run(batch_size: int = 32, num_workers: int = 16) -> None:
    """Train the model.

    Args:
        batch_size (int, optional): How many samples per batch to load. Defaults to 32.
        num_workers (int, optional): How many subprocesses to use for data loading. Defaults to 16.
    """
    dm = MNISTDataModule()
    model = PlMnist()
    trainer = pl.Trainer(
        max_epochs=100,
        gpus=torch.cuda.device_count(),
        precision=16
    )
    trainer.fit(PlMnist(), )


if __name__ == '__main__':
    run(batch_size=64, num_workers=16)
