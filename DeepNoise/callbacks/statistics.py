from typing import Dict

import torch
import wandb

from DeepNoise.builders import CALLBACKS
from DeepNoise.callbacks.base_callback import Callback


def log_stats(stats_dict: Dict[str, float], epoch: int):

    for stat_name, val in stats_dict.items():
        wandb.log({stat_name: val, "epoch": epoch})
        print(f"{stat_name} = {val}")


@CALLBACKS.register()
class SimpleStatistics(Callback):
    def __init__(self) -> None:
        self.reset()

    def reset(self):
        self.noisy_running_correct = 0
        self.noisy_running_loss = 0

        self.clean_running_correct = 0
        self.clean_running_loss = 0

        self.total_samples = 0
        self.batch_count = 0

    @torch.no_grad()
    def on_step_end(self, metrics):
        output = metrics["prediction"]
        _, predicted = torch.max(output.detach(), 1)

        self.noisy_running_loss += metrics["noisy_loss"]
        self.clean_running_loss += metrics["clean_loss"]

        self.noisy_running_correct += (predicted == metrics["noisy_label"]).sum().item()
        self.clean_running_correct += (predicted == metrics["clean_label"]).sum().item()

        self.total_samples += metrics["noisy_label"].size(0)
        self.batch_count += 1

    @torch.no_grad()
    def on_epoch_end(self, metrics):
        if self.total_samples == 0:
            return

        epoch = metrics["epoch"]
        epoch_mode = metrics["epoch_mode"]

        clean_acc = self.clean_running_correct / self.total_samples
        noisy_acc = self.noisy_running_correct / self.total_samples
        avg_clean_loss = self.clean_running_loss / self.batch_count
        avg_noisy_loss = self.noisy_running_loss / self.batch_count

        stats_dict = dict()
        stats_dict[f"{epoch_mode}_clean acc"] = clean_acc
        stats_dict[f"{epoch_mode}_noisy acc"] = noisy_acc
        stats_dict[f"{epoch_mode}_clean loss"] = avg_clean_loss
        stats_dict[f"{epoch_mode}_noisy loss"] = avg_noisy_loss
        log_stats(stats_dict, epoch)
        self.reset()
