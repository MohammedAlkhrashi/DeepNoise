import torch
from typing import List
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch.nn.functional as F
from erm import ERM
from callbacks import Callback


class SoftBootstrappingLoss(Module):
	"""
	``Loss(t, p) = - (beta * t + (1 - beta) * p) * log(p)``
	
	Args:
		beta (float): bootstrap parameter. Default, 0.95
		reduce (bool): computes mean of the loss. Default, True.
		as_pseudo_label (bool): Stop gradient propagation for the term ``(1 - beta) * p``.
			Can be interpreted as pseudo-label.
	"""
	def __init__(self, beta=0.95, reduction: 'mean' | 'sum' | None = 'mean', as_pseudo_label=True):
		super(SoftBootstrappingLoss, self).__init__()
		self.beta = beta
		self.reduction = reduction
		self.as_pseudo_label = as_pseudo_label

	def forward(self, y_pred, y):
		# cross_entropy = - t * log(p)
		beta_xentropy = self.beta * F.cross_entropy(y_pred, y, reduction='none')

		y_pred_a = y_pred.detach() if self.as_pseudo_label else y_pred
		# second term = - (1 - beta) * p * log(p)
		bootstrap = - (1.0 - self.beta) * torch.sum(F.softmax(y_pred_a, dim=1) * F.log_softmax(y_pred, dim=1), dim=1)

		if self.reduction == 'mean':
			return torch.mean(beta_xentropy + bootstrap)
		elif self.reduction == 'sum':
			return torch.sum(beta_xentropy + bootstrap)

		return beta_xentropy + bootstrap


class HardBootstrappingLoss(Module):
	"""
	``Loss(t, p) = - (beta * t + (1 - beta) * z) * log(p)``
	where ``z = argmax(p)``
	Args:
		beta (float): bootstrap parameter. Default, 0.95
		reduce (bool): computes mean of the loss. Default, True.
	"""
	def __init__(self, beta=0.8, reduction: 'mean' | 'sum' | None = 'mean'):
		super(HardBootstrappingLoss, self).__init__()
		self.beta = beta
		self.reduction = reduction

	def forward(self, y_pred, y):
		# cross_entropy = - t * log(p)
		beta_xentropy = self.beta * F.cross_entropy(y_pred, y, reduction='none')

		# z = argmax(p)
		z = F.softmax(y_pred.detach(), dim=1).argmax(dim=1)
		z = z.view(-1, 1)
		bootstrap = F.log_softmax(y_pred, dim=1).gather(1, z).view(-1)
		# second term = (1 - beta) * z * log(p)
		bootstrap = - (1.0 - self.beta) * bootstrap

		if self.reduction == 'mean':
			return torch.mean(beta_xentropy + bootstrap)
		elif self.reduction == 'sum':
			return torch.sum(beta_xentropy + bootstrap)

		return beta_xentropy + bootstrap


class BootstrappingLossTrainer(ERM):
	def __init__(
		self,
		model: Module,
		optimizer: Optimizer,
		train_loader: DataLoader,
		val_loader: DataLoader,
		test_loader: DataLoader,
		epochs: int,
		bootstrapping: 'soft' | 'hard' = 'soft',
		beta: int = 1,
		reduction: 'mean' | 'sum' | None = 'mean',
		as_pseudo_label: bool = True,
		callbacks: List[Callback] = None,
	) -> None:

		if bootstrapping == 'soft':
			loss_fn = SoftBootstrappingLoss(beta, reduction, as_pseudo_label)
		else:
			loss_fn = HardBootstrappingLoss(beta, reduction)

		
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
