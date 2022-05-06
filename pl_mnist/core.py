import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl


class PlMnist(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(28*28, 10)

    def forward(self, x):
        return F.relu(self.net(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self.net(x.view(x.size(0), -1)), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        loss = F.cross_entropy(x.view(x.size(0), -1), y)
        self.log('val_loss', loss)

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
    train_dataset = MNIST(
        root=os.getcwd(),
        train=True,
        transform=transforms.ToTensor(),
        download=False
    )
    train_dataset, val_dataset = random_split(train_dataset, [55000, 5000])
    test_dataset = MNIST(
        root=os.getcwd(),
        train=False,
        transform=transforms.ToTensor(),
        download=False
    )
    datasets = {
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset
    }
    dataloaders = {
        key: DataLoader(
            dataset=datasets[key],
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=True
        ) for key in datasets
    }
    return dataloaders


def run(batch_size: int = 32, num_workers: int = 16) -> None:
    """Train the model.

    Args:
        batch_size (int, optional): How many samples per batch to load. Defaults to 32.
        num_workers (int, optional): How many subprocesses to use for data loading. Defaults to 16.
    """
    dataloaders = make_dataloaders(batch_size, num_workers)
    trainer = pl.Trainer(
        max_epochs=100,
        gpus=2,
        precision=16
    )
    trainer.fit(
        model=PlMnist(),
        train_dataloaders=dataloaders['train'],
        val_dataloaders=dataloaders['val']
    )


if __name__ == '__main__':
    run(batch_size=64, num_workers=16)
