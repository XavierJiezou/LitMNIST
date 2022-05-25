<div align="center">

[![logo](https://raw.githubusercontent.com/XavierJiezou/LitMNIST/main/images/logo.png)](https://pixelied.com/editor/design/6282f5970515730397249959)

# LitMNIST

基于 [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) + [Hydra](https://github.com/facebookresearch/hydra) 的深度学习项目模板。

*（以 [MNIST](http://yann.lecun.com/exdb/mnist/) 分类任务为例）*

点击 [<kbd>Use this template</kbd>](https://github.com/XavierJiezou/LitMNIST/generate) 即可使用该模板来初始化你的新仓库。

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
    <a href="#演示">观看演示</a>
    •
    <a href="https://github.com/XavierJiezou/LitMNIST/issues/new">报告错误</a>
    •
    <a href="https://github.com/XavierJiezou/LitMNIST/issues/new">功能需求</a>
</p>

<p>
    <a href="/docs/README.en.md">English </a>
    •
    <a href="/docs/README.cn.md">简体中文</a>
</p>

喜欢这个项目吗？请考虑捐赠（[微信](https://raw.githubusercontent.com/XavierJiezou/ys-dl/main/image/wechat.jpg) | [支付宝](https://raw.githubusercontent.com/XavierJiezou/ys-dl/main/image/alipay.jpg)），以帮助它改善！

</div>

## 演示

![demo](https://raw.githubusercontent.com/XavierJiezou/LitMNIST/main/images/demo.jpg)

## 安装

> 开始之前，你必须熟练使用 [PyTorch Lightning](https://www.pytorchlightning.ai/)，并对 [Hydra](https://hydra.cc/) 有一定的了解。

1. 克隆仓库到本地

```bash
git clone https://github.com/XavierJiezou/LitMNIST.git
cd LitMNIST
```

2. 创建 conda 虚拟环境

```bash
conda create -n myenv python=3.7
conda activate myenv
```

3. 安装项目依赖包（如需安装 GPU 版 PyTorch，请参考[官网安装教程](https://pytorch.org/get-started/)）

```bash
pip install -r requirements.txt
```

## 结构

项目的主要目录结构如下：

```bash
├── configs # 存放 Hydra 配置文件
│   ├── callbacks # Callbacks 配置（例如 EarlyStopping、ModelCheckpoint 等）
│   ├── datamodule # Datamodule 配置（例如 batch_size、num_workers 等）
│   ├── debug # 调试配置
│   ├── experiment # 实验配置
│   ├── hparams_search # 超参数搜索配置
│   ├── local # 本地配置（暂时可以忽略）
│   ├── log_dir # 日志存放目录配置
│   ├── logger # 日志配置
│   ├── model # 模型配置
│   ├── trainer # Trainer 配置
│   │
│   ├── test.yaml # 测试的主要配置
│   └── train.yaml # 训练的主要配置
│
├── data # 存放项目数据
│
├── logs # 存放项目日志（Hydra 日志 和 PyTorch Lightning loggers 生成的日志）
│
├── src # 源代码
│   ├── datamodules # LightningDataModule
│   ├── models # 存放基于原生 PyTorch 框架编写的模型
│   ├── litmodules # LightningModule
│   ├── utils # 存放一些实用的脚本（例如数据预处理的脚本）
│   │
│   ├── testing_pipeline.py # 测试流水线
│   └── training_pipeline.py # 训练流水线
│
├── tests # 单元测试
│
├── test.py # 开始测试
├── train.py # 开始训练
│
├── .env # 存储私有环境变量（例如 wandb 的 API_KEY）【注意：该文件不受版本控制】
├── .gitignore # 设置版本控制需要排除的文件或目录
├── requirements.txt # 项目依赖环境
└── README.md # 项目概述文档
```

## 用法

### 基础用法

`train.py` 集成了模型的训练、验证及测试的一整套工作流。

```bash
python train.py
```

### 进阶用法

- 从命令行覆盖任何配置参数

`train.py` 默认从 [configs/train.yaml](configs/train.yaml) 中获取参数。因此，你可以先修改 `yaml` 配置文件中的参数，然后再运行。

当然，你也可以在命令行直接指定参数。命令行中参数的优先级要大于 `yaml` 配置文件中参数的优先级。

```bash
python train.py trainer.max_epochs=3
```

对于某些不太重要的参数，它们没有在 `train.yaml` 中定义，所以你在命令行中指定的时候必须添加 `+`，否则会报错。

```bash
python train.py +trainer.precision=16
```

- 在 CPU、GPU、多 GPU 和 TPU 上训练

```bash
# 在 CPU 上训练
python train.py trainer.gpus=0

# 在 GPU 上训练
python train.py trainer.gpus=1

# 在 TPU 上训练
python train.py +trainer.tpu_cores=8

# 基于 DDP（Distributed Data Parallel，分布式数据并行）的训练 [4 个 GPU]
python train.py trainer.gpus=4 +trainer.strategy=ddp

# 基于 DDP（Distributed Data Parallel，分布式数据并行）的训练 [8 个 GPU，两个节点]
python train.py trainer.gpus=4 +trainer.num_nodes=2 +trainer.strategy=ddp
```

- 混合精度训练

```bash
# 使用 PyTorch 本机自动混合精度（AMP）训练
python train.py trainer.gpus=1 +trainer.precision=16
```

- 使用 PyTorch Lightning 中的日志记录器来记录训练日志

```bash
# 在项目根目录下新建一个名为 `.env` 的文件，并添加一行形如 `WANDB_API_KEY="xxx"` 的文本
python train.py logger=wandb
```

- 根据自定义实验配置来训练模型

```bash
# 你可以自定义 configs/experiment/example.yaml 中的配置
python train.py experiment=example
```

- 带回调函数的训练

```python
python train.py callbacks=default
```

- 使用  Pytorch Lightning 中的策略

梯度裁剪来避免梯度爆炸

```bash
python train.py +trainer.gradient_clip_val=0.5
```

随机加权平均可以使您的模型更好地泛化

```bash
python train.py +trainer.stochastic_weight_avg=true
```

梯度累计

```bash
python train.py +trainer.accumulate_grad_batches=10
```

- 轻松调试

> 配置文件见 [configs/debug/](/configs/debug/)

默认调试模式（运行 1 个 epoch）

```bash
python train.py debug=default
```

仅对 test epoch 进行调试

```bash
python train.py debug=test_only
```

运行一个 train，val 和 test 训练（仅使用 1 个 batch）

```bash
python train.py +trainer.fast_dev_run=true
```

训练完成后打印各个阶段的执行时间（用于快速发现训练瓶颈）

```bash
python train.py +trainer.profiler="simple"
```

- 断点续训

```bash
python train.py trainer.resume_from_checkpoint="/path/to/name.ckpt"
```

- 一次执行多个实验

例如，下方代码将按顺序运行所有参数组合（共 6 个）的实验。

```bash
python train.py -m datamodule.batch_size=32,64,128 litmodule.lr=0.001,0.0005
```

此外，你也可以执行 [/configs/experiment/](/configs/experiment/) 文件夹中的的所有实验

```bash
python train.py -m 'experiment=glob(*)'
```

- 使用 Optuna 进行超参数搜索

> 配置文件见 [configs/hparams_search/](/configs/hparams_search/)

```bash
python train.py -m hparams_search=mnist_optuna
```

## Changelog

See [CHANGELOG.md](/CHANGELOG.md)

## License

[MIT License](/License)

## References

This template refers to the following warehouses and makes some modifications.

[![Readme Card](https://github-readme-stats.vercel.app/api/pin/?username=ashleve&repo=lightning-hydra-template)](https://github.com/ashleve/lightning-hydra-template)
