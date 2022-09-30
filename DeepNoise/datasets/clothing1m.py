from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.transforms import Compose


class Clothing1M(Dataset):
    def __init__(
        self,
        data_root,
        map_path,
        keys_path,
        transforms=None,
    ) -> None:
        self.paths, self.labels = self.extract_paths_and_labels(
            data_root=data_root, map_path=map_path, keys_path=keys_path
        )
        if transforms is None:
            transforms = Compose([])
        self.transforms = transforms

    def extract_paths_and_labels(self, data_root, map_path, keys_path):
        def txt_to_list(path):
            with open(path) as file:
                lines = file.readlines()
                lines = [line.rstrip() for line in lines]
            return lines

        def txt_to_dict(path):
            with open(path) as file:
                lines = file.readlines()
                lines = [line.rstrip().split() for line in lines]
            return dict(lines)

        map_path_noisy_label = txt_to_dict(f"{data_root}/annotations/{map_path}")
        paths = txt_to_list(f"{data_root}/annotations/{keys_path}")
        labels = [int(map_path_noisy_label[path]) for path in paths]
        paths = [f"{data_root}/{path}" for path in paths]
        return paths, labels

    def __getitem__(self, index):
        item = dict()

        image = Image.open(self.paths[index]).convert("RGB")
        item["image"] = image
        if self.transforms is not None:
            item["image"] = self.transforms(imagex)
        item["clean_label"] = self.labels[index]
        item["noisy_label"] = self.labels[index]
        item["sample_index"] = index
        return item

    def __len__(self):
        return len(self.labels)
