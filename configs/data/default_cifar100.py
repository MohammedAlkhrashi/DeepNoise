from configs.data.default_cifar10 import data


data["trainset"]["type"] = "NoisyCIFAR100"
data["valset"]["type"] = "NoisyCIFAR100"
data["testset"]["type"] = "NoisyCIFAR100"
