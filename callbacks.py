import torch


def log_stats(
    clean_running_correect,
    noisy_running_correct,
    clean_running_loss,
    noisy_running_loss,
    total_samples,
    batch_count,
    epoch_mode,
):
    if total_samples == 0:
        return
    clean_acc = clean_running_correect / total_samples
    noisy_acc = noisy_running_correct / total_samples

    avg_noisy_loss = noisy_running_loss / batch_count
    avg_clean_loss = clean_running_loss / batch_count

    # wandb.log({f"{epoch_mode}_clean_accuracy": clean_acc})
    # wandb.log({f"{epoch_mode}_noisy_accuracy": noisy_acc})
    # wandb.log({f"{epoch_mode}_noisy_loss": avg_loss)
    print(f"{epoch_mode}_clean acc = {clean_acc}")
    print(f"{epoch_mode}_noisy acc = {noisy_acc}")

    print(f"{epoch_mode}_clean_loss = {avg_noisy_loss}")
    print(f"{epoch_mode}_noisy_loss = {avg_clean_loss}")


class Callback:
    def on_step_end(self, metrics):
        pass

    def on_epoch_end(self, metrics):
        pass


class SimpleStats:
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
        log_stats(
            self.clean_running_correct,
            self.noisy_running_correct,
            self.clean_running_loss,
            self.noisy_running_loss,
            self.total_samples,
            self.batch_count,
            metrics["epoch_mode"],
        )
        self.reset()
