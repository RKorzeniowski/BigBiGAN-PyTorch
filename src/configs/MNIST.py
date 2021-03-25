from src.training_utils import training_utils
from src.configs import hparams
from src.configs import dataset_configs
from src.configs import utilities

config = {
    "ds_name": "MNIST",
    "num_cls": 10,
    "loading_normalization_mean": 0.5,
    "loading_normalization_var": 0.5,
    "w_init": None, # torch.nn.init.orthogonal_,
    "save_metric_interval": 10,
    "logging_interval": 40,
    **hparams.hparams,
    **dataset_configs.img_grayscale_32x32_config,
}

config = utilities.Config(**config)
