import random
import sys

import numpy as np
from scipy.stats import truncnorm
import torch


class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value


def get_config(dataset):
    config_module = "src.configs." + dataset
    __import__(config_module)
    config = sys.modules[config_module].config
    return Config(**{key: value for key, value in config.__dict__.items()
                     if not (key.startswith('__') or key.startswith('_'))})


def truncated_normal(size, threshold=1):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return torch.from_numpy(values)


def get_channel_inputs(array, input_dim=None, output_dim=None):
    if input_dim is not None:
        array = [input_dim] + list(array)
    if output_dim is not None:
        array = list(array) + [output_dim]
    return list(zip(array[:-1], array[1:]))


def set_random_seed(seed, device):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
