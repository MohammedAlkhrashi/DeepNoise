import torch
from typing import List
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch.nn.functional as F
from erm import ERM
from callbacks import Callback
import numpy as np


class BackwardCorrectedLoss(Module):
    def __init__(self, T: List[List[float]]):
        super(BackwardCorrectedLoss, self).__init__()
        self.T = T

    def forward(self, y_pred, y):
        T_inv = torch.Tensor(np.linalg.inv(self.T))
        return -torch.sum(torch.dot(y, T_inv) * torch.log(y_pred))


class ForwardCorrectedLoss(Module):
    def __init__(self, T: List[List[float]]):
        super(ForwardCorrectedLoss, self).__init__()
        self.T = T

    def forward(self, y_pred, y):
        T = torch.Tensor(self.T)
        return -torch.sum(y * torch.log(torch.dot(y_pred, T)))


class LossCorrectionTrainer(ERM):
    def __init__(
		self,
		model: Module,
		optimizer: Optimizer,
		train_loader: DataLoader,
		val_loader: DataLoader,
		test_loader: DataLoader,
		epochs: int,
		correction: str = 'backward',
        T: List[List[float]],
		callbacks: List[Callback] = None,
	) -> None:

		if correction == 'backward':
			loss_fn = BackwardCorrectedLoss(T)
		else:
			loss_fn = ForwardCorrectedLoss(T)
		
		super().__init__(
			model,
			optimizer,
			loss_fn,
			train_loader,
			val_loader,
			test_loader,
			epochs,
			callbacks,
		)
