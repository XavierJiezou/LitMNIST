from pl_mnist import __version__
import fire


class PlMnistCli:
    @staticmethod
    def version() -> str:
        """Return version of the project.

        Returns:
            str: Version of the project.
        """
        return __version__


def main() -> None:
    fire.Fire(PlMnistCli)


if __name__ == '__main__':
    main()
