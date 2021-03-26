import argparse
import itertools

from src.pipeline import pipeline
from src.training_utils import training_utils

EXP_HPARAMS = {
    "params": (
        {},
    ),
    "seeds": (420,),
}

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="FMNIST",
                    choices=["FMNIST", "MNIST", "CIFAR10", "CIFAR100", "imagenette", "imagewoof"], help="dataset name")
parser.add_argument("--data_path", type=str, default="../input/fmnist-dataset",
                    help="path to dataset root folder")
parser.add_argument("--model_architecture", type=str, default="bigbigan",
                    choices=["bigbigan", "biggan"], help="type of architecture used in training")
args = parser.parse_args()


def run_experiments():
    for hparams_overwrite_list, seed in itertools.product(EXP_HPARAMS["params"], EXP_HPARAMS["seeds"]):
        config = training_utils.get_config(args.dataset)
        hparams_str = ""
        for k, v in hparams_overwrite_list.items():
            config[k] = v
            hparams_str += str(k) + "-" + str(v) + "_"
        config["model_architecture"] = args.model_architecture
        config["hparams_str"] = hparams_str.strip("_")
        config["seed"] = seed
        run_experiment(config)


def run_experiment(config):
    training_utils.set_random_seed(seed=config.seed, device=config.device)
    if args.model_architecture == "bigbigan":
        training_pipeline = pipeline.BigBiGANPipeline.from_config(data_path=args.data_path, config=config)
    elif args.model_architecture == "bigbiwgan":
        training_pipeline = pipeline.BigBiWGANPipeline.from_config(data_path=args.data_path, config=config)
    elif args.model_architecture == "biggan":
        training_pipeline = pipeline.GANPipeline.from_config(data_path=args.data_path, config=config)
    else:
        raise ValueError(f"Architecture type {args.model_architecture} is not supported")
    training_pipeline.train_model()


run_experiments()
