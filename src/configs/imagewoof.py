from src.configs import hparams
from src.configs import dataset_configs
from src.configs import utilities

config = {
    "ds_name": "imagewoof",
    "num_cls": 10,
    "loading_normalization_mean": 0.5,
    "loading_normalization_var": 0.5,
    "w_init": None, # torch.nn.init.orthogonal_,
    "save_metric_interval": 10,
    "logging_interval": 35,
    **hparams.hparams,
    **dataset_configs.img_color_64x64_config,
}

config = utilities.Config(**config)
