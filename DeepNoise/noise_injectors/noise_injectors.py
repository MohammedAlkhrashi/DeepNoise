import numpy as np


class NoiseInjector:
    def __init__(self) -> None:
        self.T = self.get_trans_matrix()

    def apply(self, labels):
        pass

    def get_trans_matrix(self):
        raise NotImplementedError


class IdentityNoiseInjector(NoiseInjector):
    def __init__(self, num_classes) -> None:
        self.num_classes = num_classes

    def get_trans_matrix(self):
        return np.eye(self.num_classes)


class SymmetricNoiseInjector(NoiseInjector):
    def __init__(self, num_classes, noise_level) -> None:
        self.num_classes = num_classes

    def get_trans_matrix(self):
        matrix = np.eye(self.num_classes)


class AsymmetricNoiseInjector(NoiseInjector):
    def __init__(self, num_classes, noise_level) -> None:
        self.num_classes = num_classes

    def get_trans_matrix(self):
        pass


class CustomNoiseInjector(NoiseInjector):
    def __init__(self, trans_matrix) -> None:
        self.trans_matrix = trans_matrix

    def get_trans_matrix(self):
        return self.trans_matrix
