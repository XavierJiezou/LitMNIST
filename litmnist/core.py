import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from torchvision.datasets import ImageFolder
from conf import pl_config


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        persistent_workers: bool,
        pin_memory: bool
    ) -> None:
        """Define a `DataMoudle` for MNIST.

        Args:
            data_dir (str): Dir to save the dataset.
            batch_size (int): How many samples per batch to load.
            num_workers (int): How many subprocesses to use for data loading.
            persistent_workers (bool): If GPU is available, set to True.
            pin_memory (bool): If GPU is available, set to True.
        """
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
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
        if stage == 'predict' or stage is None:
            self.mnist_predict = ImageFolder(
                root='predict',
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Grayscale(1),
                    transforms.Resize((28, 28))
                ])
            )

    def train_dataloader(self) -> DataLoader:
        """For training.

        Returns:
            DataLoader: Specify training samples.
        """
        return DataLoader(
            dataset=self.mnist_train,
            batch_size=self.batch_size | self.hparams.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory
        )

    def val_dataloader(self) -> DataLoader:
        """For validation.

        Returns:
            DataLoader: Specify validation samples.
        """
        return DataLoader(
            dataset=self.mnist_val,
            batch_size=self.batch_size | self.hparams.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory
        )

    def test_dataloader(self) -> DataLoader:
        """For testing.

        Returns:
            DataLoader: Specify testing samples.
        """
        return DataLoader(
            dataset=self.mnist_test,
            batch_size=self.batch_size | self.hparams.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory
        )

    def predict_dataloader(self) -> DataLoader:
        """For predicting.

        Returns:
            DataLoader: Specify predicting samples.
        """
        return DataLoader(
            dataset=self.mnist_predict,
            batch_size=self.batch_size | self.hparams.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory
        )


class MNISTModel(nn.Module):
    def __init__(
        self,
        channels: int,
        width: int,
        height: int,
        hidden_size: int,
        num_classes: int
    ) -> None:
        """Initialization.

        Args:
            channels (int): Value is 1 for grayscale images and 3 for color images.
            width (int): Width of the image.
            height (int): Height of the image.
            hidden_size (int): Size of the hidden layers.
            num_classes (int): Number of classes to images.
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
        model: nn.Module,
        debug: bool,
        example_dims: tuple,
        learning_rate: float
    ) -> None:
        """Initialization.

        Args:
            model (nn.Module): Model defined by PyTorch.
            debug (bool): If True, display the intermediate input and output sizes of all your layers.
            example_dims (tuple): Used to create `self.example_input_array`.
            learning_rate (float): Learning rate of optimizer.
        """
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        if debug:
            self.example_input_array = torch.Tensor(*example_dims)
        else:
            pass
        self.model = model
        self.learning_rate = learning_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the prediction/inference actions.

        Args:
            x (torch.Tensor): Input.

        Returns:
            torch.Tensor: Output.
        """
        return self.model(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        """Defines the train loop. It is independent of forward.

        Args:
            batch (torch.Tensor): Input.
            batch_idx (int): Index of batch.

        Returns:
            torch.Tensor: Loss with gradient.
        """
        x, y = batch
        pred = self.model(x)
        loss = F.cross_entropy(pred, y)
        acc = accuracy(pred.argmax(1), y)
        self.log_dict({'train_loss': loss, 'train_acc': acc}, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        """Define the validation loop

        Args:
            batch (torch.Tensor): Input.
            batch_idx (int): Index of batch.
        """
        x, y = batch
        pred = self.model(x)
        loss = F.cross_entropy(pred, y)
        acc = accuracy(pred.argmax(1), y)
        self.log_dict({'val_loss': loss, 'val_acc': acc}, prog_bar=True)

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        """Define the test loop

        Args:
            batch (torch.Tensor): Input.
            batch_idx (int): Index of batch.
        """
        x, y = batch
        pred = self.model(x)
        loss = F.cross_entropy(pred, y)
        acc = accuracy(pred.argmax(1), y)
        self.log_dict({'test_loss': loss, 'test_acc': acc})

    def predict_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        """Define the predict interface.

        Args:
            batch (torch.Tensor): Current batch.
            batch_idx (int): Index of current batch.
            dataloader_idx (int): Index of the current dataloader. Default to 0.

        Returns:
            torch.Tensor: Predicted output
        """
        x, y = batch
        pred = self.model(x).argmax(1)
        return {'preds': pred, 'label': y}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Define the optimizer.

        Args:
            torch.optim.Optimizer: Optimizer.
        """
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)


def run() -> None:
    """Running.
    """
    pl.seed_everything(0, workers=True)  # For Reproducibility
    model = MNISTModel(**pl_config['model'])
    dm = MNISTDataModule(**pl_config['data'])
    lm = LitMNIST(model, **pl_config['lightning'])
    trainer = pl.Trainer(**pl_config['train'])
    if pl_config['train']['auto_scale_batch_size']:
        trainer.tune(lm, dm)
    if pl_config['train']['auto_lr_find']:
        trainer.tune(lm)
    trainer.fit(lm, dm)
    trainer.test(lm, dm)
    result = trainer.predict(lm, dm)[0]
    for key, value in result.items():
        print(key, value.tolist())


if __name__ == '__main__':
    run()
