<div align="center">

[![logo](https://raw.githubusercontent.com/XavierJiezou/LitMNIST/main/images/logo.png)](https://pixelied.com/editor/design/6282f5970515730397249959)

# LitMNIST

PyTorch Lightning template on the MNIST dataset.

<p>
    <a href="https://github.com/XavierJiezou/LitMNIST/actions?query=workflow:Release">
        <img src="https://github.com/XavierJiezou/LitMNIST/workflows/Release/badge.svg"
            alt="GitHub Workflow Release Status" />
    </a>
    <a href="https://github.com/XavierJiezou/LitMNIST/actions?query=workflow:Test">
        <img src="https://github.com/XavierJiezou/LitMNIST/workflows/Test/badge.svg"
            alt="GitHub Workflow Test Status" />
    </a>
    <a href="https://github.com/XavierJiezou/LitMNIST/actions?query=workflow:Lint">
        <img src="https://github.com/XavierJiezou/LitMNIST/workflows/Lint/badge.svg"
            alt="GitHub Workflow Lint Status" />
    </a>
    <!-- <a href='https://LitMNIST.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/LitMNIST/badge/?version=latest' alt='Documentation Status' />
    </a> -->
    <a
        href="https://www.codacy.com/gh/XavierJiezou/LitMNIST/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=XavierJiezou/LitMNIST&amp;utm_campaign=Badge_Grade">
        <img src="https://app.codacy.com/project/badge/Grade/c2f85c8d6b8a4892b40059703f087eab" alt="Codacy Badge">
    </a>
    <a href="https://codecov.io/gh/XavierJiezou/LitMNIST">
        <img src="https://codecov.io/gh/XavierJiezou/LitMNIST/branch/main/graph/badge.svg?token=QpCLcUGoYx"
            alt="codecov">
    </a>
    <a href="https://pypi.org/project/LitMNIST/">
        <img src="https://img.shields.io/pypi/v/LitMNIST" alt="PyPI">
    </a>
    <a href="https://pypistats.org/packages/LitMNIST">
        <img src="https://img.shields.io/pypi/dm/LitMNIST" alt="PyPI - Downloads">
    </a>
    <!-- <a href="https://pypi.org/project/LitMNIST/">
        <img src="https://img.shields.io/pypi/pyversions/LitMNIST" alt="PyPI - Python Version">
    </a> -->
    <a href="https://github.com/XavierJiezou/LitMNIST/stargazers">
        <img src="https://img.shields.io/github/stars/XavierJiezou/LitMNIST" alt="GitHub stars">
    </a>
    <a href="https://github.com/XavierJiezou/LitMNIST/network">
        <img src="https://img.shields.io/github/forks/XavierJiezou/LitMNIST" alt="GitHub forks">
    </a>
    <a href="https://github.com/XavierJiezou/LitMNIST/issues">
        <img src="https://img.shields.io/github/issues/XavierJiezou/LitMNIST" alt="GitHub issues">
    </a>
    <a href="https://github.com/XavierJiezou/LitMNIST/blob/main/LICENSE">
        <img src="https://img.shields.io/github/license/XavierJiezou/LitMNIST" alt="GitHub license">
    </a>
    <!-- <a href="https://github.com/psf/black">
        <img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg" />
    </a> -->
</p>

<p>
    <!-- <a href="https://www.python.org/">
        <img src="http://ForTheBadge.com/images/badges/made-with-python.svg" alt="forthebadge made-with-python">
    </a>
    <a href="https://github.com/XavierJiezou">
        <img src="http://ForTheBadge.com/images/badges/built-with-love.svg" alt="ForTheBadge built-with-love">
    </a> -->
    <a href="https://www.python.org/">
        <img alt="Python" src="https://img.shields.io/badge/-Python 3.7+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
    <a href="https://pytorch.org/get-started/locally/">
        <img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.8+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
    <a href="https://pytorchlightning.ai/">
        <img alt="Lightning" src="https://img.shields.io/badge/-Lightning 1.5+-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white">
    </a>
    <a href="https://hydra.cc/">
        <img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra 1.1-89b8cd?style=for-the-badge&labelColor=gray"></a>
    <a href="https://black.readthedocs.io/en/stable/">
        <img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray">
    </a>
</p>

<p>
    <a href="#demo">View Demo</a>
    •
    <a href="https://github.com/XavierJiezou/LitMNIST/issues/new">Report Bug</a>
    •
    <a href="https://github.com/XavierJiezou/LitMNIST/issues/new">Request Feature</a>
</p>

<p>
    <a href="/docs/README.en.md">English </a>
    •
    <a href="/docs/README.cn.md">简体中文</a>
</p>

Love the project? Please consider [donating](https://paypal.me/xavierjiezou?country.x=C2&locale.x=zh_XC) to help it improve!

</div>

## Demo

![demo](https://raw.githubusercontent.com/XavierJiezou/LitMNIST/main/images/demo.jpg)

## Quickstart

```bash
# clone project
git clone https://github.com/XavierJiezou/LitMNIST.git
cd LitMNIST

# [OPTIONAL] create conda environment
conda create -n myenv python=3.7
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

## Usage

### Train

```bash
python train
```

### Test

## Usage

`$ litmnist`

## Changelog

See [CHANGELOG.md](/CHANGELOG.md)

## License

[MIT License](/License)

## Dependencies

### Production Dependencies

[![Readme Card](https://github-readme-stats.vercel.app/api/pin/?username=psf&repo=requests)](https://github.com/psf/requests)

[![Readme Card](https://github-readme-stats.vercel.app/api/pin/?username=Textualize&repo=rich)](https://github.com/Textualize/rich)

[![Readme Card](https://github-readme-stats.vercel.app/api/pin/?username=google&repo=python-fire)](https://github.com/google/python-fire)

### Development dependencies

[![Readme Card](https://github-readme-stats.vercel.app/api/pin/?username=python-poetry&repo=poetry)](https://github.com/python-poetry/poetry)

[![Readme Card](https://github-readme-stats.vercel.app/api/pin/?username=pytest-dev&repo=pytest)](https://github.com/pytest-dev/pytest)

[![Readme Card](https://github-readme-stats.vercel.app/api/pin/?username=pytest-dev&repo=pytest-cov)](https://github.com/pytest-dev/pytest-cov)

[![Readme Card](https://github-readme-stats.vercel.app/api/pin/?username=pre-commit&repo=pre-commit)](https://github.com/pre-commit/pre-commit)

[![Readme Card](https://github-readme-stats.vercel.app/api/pin/?username=PyCQA&repo=flake8)](https://github.com/PyCQA/flake8)

[![Readme Card](https://github-readme-stats.vercel.app/api/pin/?username=PyCQA&repo=pylint)](https://github.com/PyCQA/pylint)

[![Readme Card](https://github-readme-stats.vercel.app/api/pin/?username=psf&repo=black)](https://github.com/psf/black)

[![Readme Card](https://github-readme-stats.vercel.app/api/pin/?username=uiri&repo=toml)](https://github.com/uiri/toml)

[![GitHub issues](https://img.shields.io/github/issues/XavierJiezou/LitMNIST)](https://github.com/XavierJiezou/LitMNIST/issues)
[![GitHub license](https://img.shields.io/github/license/XavierJiezou/LitMNIST)](https://github.com/XavierJiezou/LitMNIST/blob/main/LICENSE)

## References

- [Python dependency management and packaging made easy.](https://github.com/python-poetry/poetry)
- [The lightweight PyTorch wrapper for high-performance AI research. Scale your models, not the boilerplate.](https://github.com/PyTorchLightning/pytorch-lightning)
- [PyTorch Lightning + Hydra. A very user-friendly template for rapid and reproducible ML experimentation with best practices.](https://github.com/ashleve/lightning-hydra-template)
- - [一文详解 PyTorch 中的交叉熵](https://zhuanlan.zhihu.com/p/369699003)
