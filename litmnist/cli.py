"""Ref
https://pytorch-lightning.readthedocs.io/en/latest/cli/lightning_cli_intermediate.html#enable-the-lightning-cli
https://pytorch-lightning.readthedocs.io/en/latest/cli/lightning_cli_intermediate_2.html
"""

from litmnist import __version__
from litmnist.core import MNISTDataModule, LitMNIST
from pytorch_lightning.utilities.cli import LightningCLI
import fire


class LitMNISTCLI:
    @staticmethod
    def version() -> str:
        """Return version of the project.

        Returns:
            str: Version of the project.
        """
        return __version__


def main() -> None:
    fire.Fire(LitMNISTCLI)
    # LightningCLI(LitMNIST, MNISTDataModule)


if __name__ == '__main__':
    main()
