import os.path as osp
from copy import copy
from typing import Dict

import numpy as np
import torch

from DeepNoise.builders import NOISE_INJECTORS


def load_from_path(path: str):
    _, ext = osp.splitext(path)
    if "pt" in ext:
        return torch.load(path)
    elif "np" in ext:
        return np.load(path)
    else:
        raise NotImplementedError(f"Files with {ext} extension are not supported.")


class NoiseInjector:
    """
    Noise injection interface, classes subclassing this class should
    implement the method create_noise_transition_matrix.
    """

    def apply(self, labels, num_classes: int = None) -> np.array:
        """
        Returns noisy labels by flipping some of the given labels probabilisticly
        based on the noise transition matrix.

        Args:
            labels (iterable),
            num_classes (int, optional): the number of classes in the dataset, if None
            the number of classes will be inferred from the labels.

        Returns:
            np.array: the noisy labels
        """
        labels = np.array(labels, dtype=int)
        classes = np.unique(labels)
        if num_classes is None:
            num_classes = len(classes)
        if num_classes <= 1:
            raise ValueError(
                f"num_classes must be greater than 1, but num_classes = {num_classes}"
            )
        t_matrix = self.create_noise_transition_matrix(num_classes)

        noisy_labels = np.copy(labels)
        for i, label in enumerate(labels):
            noise_transition_row = t_matrix[label]
            noisy_labels[i] = np.random.choice(classes, p=noise_transition_row)

        return noisy_labels

    def create_noise_transition_matrix(self, num_classes):
        raise NotImplementedError


@NOISE_INJECTORS.register("SymmetricNoise")
class SymmetricNoiseInjector(NoiseInjector):
    def __init__(self, noise_prob: float, allow_equal_flips: bool = True) -> None:
        """Handles injecting symmetric noise.

        Args:
            noise_prob (float): The fraction of labels to be changed per class.
            allow_equal_flips (bool, optional): When this is True, allow for the possibility
            of some labels being randomly flipped to the same value, i.e., the random flip can results in
            the label not changing).

        """
        if noise_prob < 0 or noise_prob > 1:
            raise ValueError(f"noise_prob should be between 0 and 1 (inclusive)")

        self.noise_prob = noise_prob
        self.allow_equal_flips = allow_equal_flips

    def create_noise_transition_matrix(self, num_classes):
        I = np.eye(num_classes)
        if not self.allow_equal_flips:
            noise_prob = self.noise_prob - (self.noise_prob) / num_classes
        else:
            noise_prob = self.noise_prob
        off_diag_prob = (noise_prob) / (num_classes - 1)

        diag_matrix = (1 - noise_prob) * I
        full_matrix = np.full((num_classes, num_classes), fill_value=off_diag_prob)
        off_diag_matirx = full_matrix - off_diag_prob * I
        t_matrix = diag_matrix + off_diag_matirx
        assert np.allclose((np.sum(t_matrix, axis=0)), 1)
        assert np.allclose((np.sum(t_matrix, axis=1)), 1)
        return t_matrix


@NOISE_INJECTORS.register("AsymmetricNoise")
class AsymmetricNoiseInjector(NoiseInjector):
    def __init__(
        self,
        noise_prob: float,
        noise_map: Dict = None,
    ) -> None:
        """Handles injecting assymmetric noise to clean labels.

        Args:
            noise_prob (float): The fraction of labels to be changed per class.
            noise_map (Dict, optional): a dictionary that indicates what each class is
            flipped to. If noise_map is None, then each class is changed to the next class
            cyclically.
        """
        self.noise_prob = noise_prob
        self.noise_map = noise_map

    def create_noise_transition_matrix(self, num_classes):
        if self.noise_map is None:
            noise_map = {i: (i + 1) % num_classes for i in range(num_classes)}
        else:
            noise_map = self.noise_map

        t_matrix = np.eye(num_classes)
        for row_idx in range(num_classes):
            from_class = row_idx
            to_class = noise_map[from_class]
            t_matrix[row_idx][from_class] -= self.noise_prob
            t_matrix[row_idx][to_class] += self.noise_prob

        assert np.allclose((np.sum(t_matrix, axis=0)), 1)
        assert np.allclose((np.sum(t_matrix, axis=1)), 1)
        return t_matrix


@NOISE_INJECTORS.register("CustomMatrixNoiseInjector")
class CustomMatrixNoiseInjector(NoiseInjector):
    def __init__(self, transition_matrix) -> None:
        """
        transition_matrix (iterable[iterable]): The transition matrix that will be used
        to geenrate the noisy labels, if str it will be treated as a file path.
        """

        if isinstance(transition_matrix, str):
            transition_matrix = load_from_path(transition_matrix)

        transition_matrix = np.array(transition_matrix)
        if not (
            transition_matrix.ndim == 2
            and transition_matrix.shape[0] == transition_matrix.shape[1]
        ):
            raise ValueError("transition_matrix must me a square matrix.")

        if not (
            np.allclose((np.sum(transition_matrix, axis=0)), 1)
            and np.allclose((np.sum(transition_matrix, axis=1)), 1)
        ):
            raise ValueError(
                "Rows and columns of the transition matrix must sum to one."
            )

        self.transition_matrix = transition_matrix

    def create_noise_transition_matrix(self, num_classes):
        if num_classes > self.transition_matrix.shape[0]:
            raise ValueError(
                "Number of classes cannot be larger than the number"
                " of rows of the transition matrix"
            )

        return self.transition_matrix


@NOISE_INJECTORS.register("CustomLabelsNoiseInjector")
class CustomLabelsNoiseInjector(NoiseInjector):
    def __init__(self, noisy_labels) -> None:
        """
        Args:
            noisy_labels (iterable | str): The noisy labels that will replace the clean labels.
            If str it will be treated as a file path.
        """
        super().__init__()
        if isinstance(noisy_labels, str):
            noisy_labels = load_from_path(noisy_labels)

        self.noisy_labels = np.array(noisy_labels)

    def apply(self, labels, num_classes: int = None) -> np.array:
        if len(labels) != len(self.noisy_labels):
            raise ValueError(
                f"len of the given labels ({len(labels)}) must equal the len"
                f" of the noisy labels ({len(self.noisy_labels)})"
            )
        return self.noisy_labels


@NOISE_INJECTORS.register("IdentityNoise")
class IdentityNoiseInjector(NoiseInjector):
    def create_noise_transition_matrix(self, num_classes):
        return np.eye(num_classes)
