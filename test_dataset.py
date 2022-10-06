from torch.utils.data import DataLoader

from DeepNoise.datasets.cifar import NoisyCIFAR10
from DeepNoise.noise_injectors import SymmetricNoiseInjector

train = NoisyCIFAR10(
    noise_injector=SymmetricNoiseInjector(0.2),
    root="data",
    train=True,
    download=True,
)
test = NoisyCIFAR10(
    root="data",
    train=False,
    download=True,
)

print(train[0])
