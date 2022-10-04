from configs.algorithms.default_erm import cfg

cfg["trainer"] = dict()
cfg["trainer"]["type"] = "BootStrapping"
cfg["trainer"]["bootstrapping"] = "hard"
cfg["trainer"]["ignore_passed_loss_fn"] = True
cfg["trainer"]["beta"] = 0.8
