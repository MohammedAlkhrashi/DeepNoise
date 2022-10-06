import numpy as np
from sklearn.metrics import confusion_matrix

from DeepNoise.noise_injectors import (
    AsymmetricNoiseInjector,
    CustomNoiseInjector,
    SymmetricNoiseInjector,
)

if __name__ == "__main__":
    # Temporary testing location

    CustomNoiseInjector(SymmetricNoiseInjector(0.4).create_noise_transition_matrix(23))
    CustomNoiseInjector(AsymmetricNoiseInjector(0.4).create_noise_transition_matrix(44))

    labels = [[i] * 100 for i in range(10)]
    labels = np.array(labels).flatten()
    total_cm = np.zeros((10, 10))
    trails = 10
    for i in range(trails):
        noise_injector = SymmetricNoiseInjector(noise_prob=0.6, allow_equal_flips=True)
        noisy_labels = noise_injector.apply(labels)
        total_cm += confusion_matrix(labels, noisy_labels) / 100

    print(total_cm / trails)
