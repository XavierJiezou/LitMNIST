"""Ref
https://pytorch-lightning.readthedocs.io/en/latest/cli/lightning_cli_intermediate.html#enable-the-lightning-cli
https://pytorch-lightning.readthedocs.io/en/latest/cli/lightning_cli_intermediate_2.html
"""


import fire

from litmnist import __version__


class LitMNISTCli:
    @property
    def version(self) -> str:
        """Return version of the project.

        Returns:
            str: Version of the project.
        """
        return __version__


def main() -> None:
    fire.Fire(LitMNISTCli)


if __name__ == "__main__":
    main()
