import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from config import pl_config


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        dims: tuple,
        num_classes: int,
        batch_size: int,
        num_workders: int
    ) -> None:
        """Define a `DataMoudle` for MNIST.

        Args:
            data_dir (str): Dir to save the dataset.
            dims (tuple): Dimension of the input image. Format in (channels, width, height).
            num_classes (int): Number of classes to images.
            batch_size (int): How many samples per batch to load.
            num_workers (int): How many subprocesses to use for data loading.
        """
        super().__init__()
        self.data_dir = data_dir
        self.dims = dims
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def prepare_data(self) -> None:
        """Use this to download and prepare data.
        """
        MNIST(self.data_dir, train=True, download=not os.path.exists('MNIST'))
        MNIST(self.data_dir, train=False, download=not os.path.exists('MNIST'))

    def setup(self, stage: str = None) -> None:
        """Called at the beginning of fit (train + validate), validate, test, or predict.

        Args:
            stage (str, optional): Either 'fit', 'validate', 'test', or 'predict'. Defaults to None.
        """
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

    def train_dataloader(self) -> DataLoader:
        """Implement one or more PyTorch DataLoaders for training.

        Returns:
            DataLoader: A collection of torch.utils.data.DataLoader specifying training samples.
        """
        return DataLoader(
            dataset=self.mnist_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def val_dataloader(self) -> DataLoader:
        """Implement one or more PyTorch DataLoaders for validation.

        Returns:
            DataLoader: A collection of torch.utils.data.DataLoader specifying validation samples.
        """
        return DataLoader(
            dataset=self.mnist_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def test_dataloader(self) -> DataLoader:
        """Implement one or more PyTorch DataLoaders for testing.

        Returns:
            DataLoader: A collection of torch.utils.data.DataLoader specifying testing samples.
        """
        return DataLoader(
            dataset=self.mnist_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True
        )


class MNISTModel(nn.Module):
    def __init__(
        self,
        channels: int,
        width: int,
        height: int,
        hidden_size: int
    ) -> None:
        """Initialization.

        Args:
            channels (int): Value is 1 for grayscale images and 3 for color images.
            width (int): Width of the image.
            height (int): Height of the image.
            hidden_size (int): Size of the hidden layers.
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            x (torch.Tensor): Input.

        Returns:
            torch.Tensor: Ouput.
        """
        return self.model(x)


class LitMNIST(pl.LightningModule):
    def __init__(
        self,
        debug: bool,
        example_dims: tuple,
        learning_rate: float
    ):
        """_summary_

        Args:
            debug (bool): If True, 
            example_dims (tuple): _description_
            learning_rate (float): _description_
        """
        super().__init__()
        if debug:
            self.example_input_array = torch.Tensor(*example_dims)
        else:
            pass
        self.model = MNISTModel(**pl_config['model'])
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = F.cross_entropy(pred, y)
        acc = accuracy(pred.argmax(1), y)
        self.log_dict({'train_loss': loss, 'train_acc': acc})
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        pred = self.model(x)
        loss = F.cross_entropy(pred, y)
        acc = accuracy(pred.argmax(1), y)
        self.log_dict({'val_loss': loss, 'val_acc': acc})

    def test_step(self, test_batch, batch_idx):
        self.validation_step(test_batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def run() -> None:
    """Train the model.
    """
    dm = MNISTDataModule()
    model = LitMNIST()
    trainer = pl.Trainer(
        max_epochs=100,
        gpus=torch.cuda.device_count(),
        precision=16,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor='val_loss',
                mode='min',
                patience=10
            ),
            pl.callbacks.DeviceStatsMonitor()
        ],
    )
    trainer.fit(PlMnist(), )


if __name__ == '__main__':
    run()
