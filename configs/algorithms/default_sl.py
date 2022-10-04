from configs.algorithms.default_erm import cfg

cfg["trainer"] = dict()
cfg["trainer"]["type"] = "SymmetericLoss"
cfg["trainer"]["ignore_passed_loss_fn"] = True
cfg["trainer"]["alpha"] = 0.1
cfg["trainer"]["beta"] = 1
cfg["trainer"]["A"] = -4
