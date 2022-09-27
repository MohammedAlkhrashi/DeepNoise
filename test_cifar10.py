import timm
from torch.optim import SGD
import torchvision.transforms as T
import torch.nn as nn
import numpy as np
from dataset import NoisyDataset, create_cifar10_dataset
from callbacks import Callback, SimpleStats
import numpy as np

from erm import ERM

from torch.utils.data import DataLoader
from pencil import Pencil

from symmetric_loss import SymmetericLossTrainer
from bootstrap import BootstrappingLossTrainer


def test_all():
    model: nn.Module = timm.create_model("efficientnet_b0", pretrained=False, num_classes=10)
    optim = SGD(model.parameters(), 0.02,momentum=0.9,weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()

    train_loader,val_loader,test_loader = create_cifar10_dataset()
    
    callbacks = [SimpleStats()]

    print("START ERM")
    trainer = ERM(
        model=model,
        optimizer=optim,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=1,
        callbacks=callbacks,
    )
    trainer.start()

    print("START Symmetric")

    trainer = SymmetericLossTrainer(
        model=model,
        optimizer=optim,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=1,
        callbacks=callbacks,
    )
    trainer.start()




    print("Start Pencil")
    trainer = Pencil(
        model=model,
        optimizer=optim,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=1,
        callbacks=callbacks,
        training_labels=train_loader.dataset.noisy_labels,
        alpha=0.01,
        b=0.1,
        labels_lr=500,
        stages=[70,200,320]
    )
    trainer.start()



    

if __name__ == "__main__":
    test_all()
    