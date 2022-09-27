from torch.utils.data import Dataset


class CIFAR10(Dataset):
    def __init__(self) -> None:
        super().__init__()
