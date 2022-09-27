from DeepNoise.builders.registry import Registry

TRAINERS = Registry()
DATASET = Registry()
MODELS = Registry()


def build_dataset(cfg):
    pass

def build_loaders(cfg):
    pass


def build_model(cfg):
    pass


def build_trainer(cfg):
    pass


def build_optimizer(cfg):
    pass


def build_loss(cfg):
    pass

def build_callbacks(cfg):
    pass
