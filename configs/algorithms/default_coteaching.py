from configs.algorithms.default_erm import cfg

cfg["optimizer"] = dict()
cfg["optimizer"]["type"] = "CoTeachingOptimizers"
cfg["optimizer"]["optim_1_cfg"] = dict()
cfg["optimizer"]["optim_2_cfg"] = dict()

cfg["optimizer"]["optim_1_cfg"]["type"] = "SGD"
cfg["optimizer"]["optim_1_cfg"]["lr"] = 0.02
cfg["optimizer"]["optim_1_cfg"]["weight_decay"] = 0.0005
cfg["optimizer"]["optim_1_cfg"]["momentum"] = 0.9

cfg["optimizer"]["optim_2_cfg"]["type"] = "SGD"
cfg["optimizer"]["optim_2_cfg"]["lr"] = 0.02
cfg["optimizer"]["optim_2_cfg"]["weight_decay"] = 0.0005
cfg["optimizer"]["optim_2_cfg"]["momentum"] = 0.9

cfg["trainer"] = dict()
cfg["trainer"]["type"] = "Co-Teaching"
cfg["trainer"]["forget_rate"] = 0.2
cfg["trainer"]["num_gradual"] = 10
cfg["trainer"]["exponent"] = 1
