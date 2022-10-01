import wandb
from DeepNoise.builders.builders import CALLBACKS
from torch.optim.lr_scheduler import MultiStepLR
from DeepNoise.callbacks import Callback


@CALLBACKS.register("StepLR")
class StepLRScheduler(Callback):
    def __init__(self, optimizer, milestones, gamma, last_epoch=-1):
        self.scheduler = MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma, last_epoch=last_epoch
        )

    def on_step_end(self, metrics):
        pass

    def on_epoch_end(self, metrics):
        epoch_mode = metrics["epoch_mode"]
        epoch = metrics["epoch"]
        if epoch_mode != "train":
            return

        self.scheduler.step()

        cur_lr = self.scheduler.get_last_lr()[0]
        print(f"Current LR is: {cur_lr}")
        wandb.log({"learning_rate": cur_lr, "epoch": epoch})

    def on_training_end(self, metrics):
        pass
