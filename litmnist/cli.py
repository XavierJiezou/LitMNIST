from litmnist import __version__
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


if __name__ == '__main__':
    main()
