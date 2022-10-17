<h1 align="center">DeepNoise</h1>

 <p align="center">
    A PyTorch framework for learning with noisy labels.
    <br />
    <a href="https://github.com/MohammedAlkhrashi/DeepNoise/issues">Report Bug</a>
    ·
    <a href="https://github.com/MohammedAlkhrashi/DeepNoise/issues">Request Feature</a>
  </p>

<div align="center">

</div>

<hr />

<p align="center">
Deep noise is a PyTorch framework for learning with noisy labels in the context of deep learning. It provides implementations for well-known algorithms, common datasets in the literature, and noise-related utilities for learning from noisy labels.

</p>

Note that the DeepNoise framework is in an early stage of development. We plan to implement more algorithms, datasets, better logging and visualizations, and more noise-related utilities. The current API and structure of the framework might change in the future. All suggestions and contributions are welcome.

## Built with

- [PyTorch](https://pytorch.org/)

## Getting Started

### Installation

```bash
git clone https://github.com/MohammedAlkhrashi/DeepNoise.git
cd DeepNoise
pip install -r requriments.txt
```

### Usage

### 1. Define a noise injection strategy

```python
from DeepNoise.noise_injectors import SymmetricNoiseInjector
noise_injector = SymmetricNoiseInjector(noise_prob=0.4)
```

### 2. Define a torch dataset

```python
from DeepNoise.datasets.cifar import NoisyCIFAR10
train_set = NoisyCIFAR10(
    noise_injector=noise_injector,
    train=True,
    download=True,
    transforms=train_transform,
    root=data_root,
)
test_set = NoisyCIFAR10(
    train=False, download=True, transforms=test_transforms, root=data_root
)
```

### 3. Define callbacks and initilizae wandb

```python
from DeepNoise.callbacks.lr_scheduler import StepLRScheduler
from DeepNoise.callbacks.statistics import SimpleStatistics
callbacks = [
    SimpleStatistics(),
    StepLRScheduler(optimizer, milestones=[80, 120], gamma=0.1),
]

wandb.init(project="DeepNoise")
```

### 4. Define a trainer

```python
from DeepNoise.algorithms.symmetric_loss import SymmetericLossTrainer
trainer = SymmetericLossTrainer(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=test_loader,
    epochs=epochs,
    callbacks=callbacks,
)
```

#### 5. Finally, train on your noisy data using a robust algorithm!

```python
trainer.start()
```

Check [example.py](https://github.com/MohammedAlkhrashi/DeepNoise/blob/main/example.py) for a full running example.

## Todo

- [ ] Uplaod DeepNoise to PyPI
- [ ] Add a documentation website
- [ ] Improve Logging
- [ ] Add resouces used

## Authors

- [@MohammedAlkhrashi](https://github.com/MohammedAlkhrashi)
- [@HishamAlyahya](https://github.com/HishamYahya)

## Acknowledgement

- [README template by @DeeshanSharma](https://github.com/DeeshanSharma/readme-template)
