import sys

sys.path.append(".")

from torchvision.utils import save_image

from litmnist.datamodules import MNISTDataModule


def make_logo(save_path: str) -> None:
    datamodule = MNISTDataModule()
    datamodule.prepare_data()
    datamodule.setup()
    temp = {i: [] for i in range(10)}
    for x, y in datamodule.data_test:
        if len(temp[y]) < 10:
            temp[y] += x
        else:
            if sum(len(value) for value in temp.values()) == 100:
                break
            else:
                continue
    logo = [each.unsqueeze(0) for value in temp.values() for each in value]
    save_image(logo, save_path, nrow=10, padding=0)


if __name__ == "__main__":
    make_logo("images/logo.png")
