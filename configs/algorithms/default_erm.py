from configs.data.default_cifar10 import data

cfg = dict()
cfg["data"] = data
cfg["data"]["trainset"]["noise_injector"] = dict(type="SymmetricNoise", noise_prob=0.4)

cfg["batch_size"] = 2
cfg["num_classes"] = 10
cfg["num_workers"] = 2
cfg["epochs"] = 5

cfg["model"] = dict()
cfg["model"]["type"] = "resnet34"
cfg["model"]["pretrained"] = False

cfg["optimizer"] = dict()
cfg["optimizer"]["type"] = "SGD"
cfg["optimizer"]["lr"] = 0.02
cfg["optimizer"]["weight_decay"] = 0.0005
cfg["optimizer"]["momentum"] = 0.9

cfg["loss_fn"] = dict()
cfg["loss_fn"]["type"] = "CrossEntropyLoss"

cfg["callbacks"] = [dict(type="SimpleStatistics")]
cfg["optimizer_callbacks"] = [
    dict(type="StepLR", milestones=[80, 100], gamma=0.1, last_epoch=-1)
]


cfg["trainer"] = dict()
cfg["trainer"]["type"] = "ERM"
