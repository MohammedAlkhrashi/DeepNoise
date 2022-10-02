data = dict()

data["trainset"] = dict()
data["trainset"]["type"] = "NoisyCIFAR10"
data["trainset"]["train"] = True
data["trainset"]["root"] = "data"
data["trainset"]["download"] = True
data["trainset"]["transforms"] = [
    dict(type="RandomCrop", size=32, padding=4, padding_mode="reflect"),
    dict(type="RandomHorizontalFlip"),
    dict(type="ToTensor"),
    dict(
        type="Normalize",
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
    ),
]

data["valset"] = dict()
data["valset"]["type"] = "NoisyCIFAR10"
data["valset"]["train"] = False
data["valset"]["root"] = "data"
data["valset"]["download"] = True
data["valset"]["transforms"] = [
    dict(type="ToTensor"),
    dict(
        type="Normalize",
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
    ),
]

data["testset"] = dict()
data["testset"]["type"] = "NoisyCIFAR10"
data["testset"]["train"] = False
data["testset"]["root"] = "data"
data["testset"]["download"] = True
data["testset"]["transforms"] = [
    dict(type="ToTensor"),
    dict(
        type="Normalize",
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
    ),
]

# Data End
