from configs.algorithms.default_erm import cfg

cfg["trainer"] = dict()
cfg["trainer"]["type"] = "Pencil"
cfg["trainer"]["labels_lr"] = 400
cfg["trainer"]["alpha"] = 0.1
cfg["trainer"]["beta"] = 0.8
cfg["trainer"]["stages"] = [70, 200, 250]
cfg["trainer"]["ignored_passed_loss_fn"] = True
cfg["trainer"]["num_classes"] = cfg["num_classes"]
cfg["trainer"]["training_labels"] = "noisy_labels"
cfg["trainer"]["ignore_passed_loss_fn"] = True

cfg["epochs"] = 250
