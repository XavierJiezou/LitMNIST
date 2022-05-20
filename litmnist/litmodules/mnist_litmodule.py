from typing import Any

import torch
from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy


class MNISTLitModule(LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.model = model

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _step(self, batch: Any) -> tuple[Any]:
        x, y = batch
        out = self.model(x)
        preds = out.argmax(1)
        loss = self.criterion(out, y)
        acc = accuracy(preds, y)
        return preds, loss, acc

    def training_step(self, batch: Any) -> Any:
        _, loss, acc = self._step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        _, loss, acc = self._step(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch: Any) -> Any:
        _, loss, acc = self._step(batch)
        metrics = {"test_loss": loss, "test_acc": acc}
        self.log_dict(metrics)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
