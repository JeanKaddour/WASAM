import argparse
import dataclasses
from typing import List

from models import MODELS


@dataclasses.dataclass
class Config:
    # ------------------------------- EXPERIMENT -------------------------------------
    no_cuda: bool
    seed: int
    device: int
    dataset_path: str
    dataset_name: str
    # ------------------------------- LOGGING -------------------------------------
    log_interval: int
    log_filepath: str
    log_to_wandb: bool
    # ------------------------------- MODEL -------------------------------------
    model: str
    dropout: float
    # ------------------------------- OPTIMIZER -------------------------------------
    optimizer_name: str
    pretrained: bool
    num_workers: int
    batch_size: int
    num_classes: int
    loss: str
    label_smoothing_factor: float
    output_path: str


@dataclasses.dataclass
class TrainConfig(Config):
    use_val_set: bool
    validate_best_model_saved: bool
    val_interval: int
    val_size: float
    # ------------------------------- SAM -------------------------------------
    sam_rho: float
    # ------------------------------- SWA -------------------------------------
    is_swa: bool
    swa_starts: List[int]
    swa_lr: float
    swa_scheduler: bool
    # ------------------------------- BASE OPTIMIZER -------------------------------------
    max_epochs: int
    save_epoch_interval: int
    base_optimizer_name: str
    learning_rate: float
    lr_scheduler: str
    lr_gamma: float
    adam_beta_1: float
    adam_beta_2: float
    weight_decay: float
    sgd_momentum: float
    sgd_nesterov: bool


def str2bool(v: str) -> bool:
    return v.lower() == "true"


def get_parser() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser()
    # ------------------------------- LOGGING -------------------------------------
    argparser.add_argument("--log_filepath", type=str, default="./log/")
    argparser.add_argument("--log_to_wandb", type=str2bool, default=True)
    argparser.add_argument(
        "--log_interval", default=20, type=int, help="Log every n steps"
    )
    # ------------------------------- MODEL -------------------------------------
    argparser.add_argument(
        "--model",
        default="resnet18",
        choices=MODELS,
        help="model architecture: " + " (default: resnet50)",
    )
    argparser.add_argument("--pretrained", default=False, type=str2bool)
    # ------------------------------- REGULARIZATION -------------------------------------
    argparser.add_argument("--weight_decay", type=float, default=0.0005)
    argparser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    # ------------------------------- EXPERIMENT -------------------------------------
    argparser.add_argument(
        "--num_workers",
        default=8,
        type=int,
        help="number of data loading workers",
    )
    argparser.add_argument("--seed", type=int, default=1)
    argparser.add_argument(
        "--no_cuda", type=str2bool, default=False, help="Disable CUDA"
    )
    argparser.add_argument("--device", type=int, default=0)
    argparser.add_argument(
        "--dataset_name",
        default="cifar100",
        help="dataset name",
        choices=["cifar10", "cifar100", "imagenet", "svhn"],
    )
    argparser.add_argument(
        "--dataset_path",
        default="./datasets/",
        type=str,
        help="path to dataset",
    )
    argparser.add_argument(
        "--output_path",
        default=f"./saved_models/",
        type=str,
        help="path to save output (e.g. model or plots)",
    )
    argparser.add_argument(
        "--save_epoch_interval",
        default=100,
        type=int,
        help="path to dataset",
    )
    argparser.add_argument("--use_val_set", type=str2bool, default=True)
    argparser.add_argument(
        "--validate_best_model_saved",
        default=True,
        type=str2bool,
        help="If instead of validate on last model state; load best model saved for test evaluation",
    )
    argparser.add_argument(
        "--val_size",
        default=0.1,
        type=float,
        help="Size of validation set",
    )
    argparser.add_argument(
        "--val_interval",
        default=1,
        type=int,
        help="Evaluate validation set every n steps",
    )

    # ------------------------------- OPTIMIZATION -------------------------------------
    argparser.add_argument(
        "--batch_size",
        default=256,
        type=int,
        help="mini-batch size (default: 256), this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel",
    )
    argparser.add_argument(
        "--optimizer_name",
        default="sgd",
        type=str,
        choices=[
            "adam",
            "adamW",
            "sam",
            "swa",
            "asam",
            "sgd",
        ],
    )
    argparser.add_argument(
        "--base_optimizer_name", type=str, default="sgd", choices=["sgd"]
    )
    argparser.add_argument(
        "--loss",
        type=str,
        default="label_smoothing",
        choices=["cross_entropy", "label_smoothing"],
    )
    argparser.add_argument("--label_smoothing_factor", type=float, default=1e-1)
    argparser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-1,
        help="Base learning rate at the start of the training.",
    )
    argparser.add_argument("--adam_beta_1", type=float, default=0.9)
    argparser.add_argument("--adam_beta_2", type=float, default=0.999)
    argparser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=[
            "none",
            "cosine",
            "multistep",
            "warmup_cosine",
            "warmup_linear",
            "warmup_constant",
        ],
    )
    argparser.add_argument("--lr_gamma", type=float, default=0.9)
    argparser.add_argument(
        "--sgd_momentum", default=0.9, type=float, help="SGD Momentum."
    )
    argparser.add_argument("--sgd_nesterov", type=str2bool, default=False)
    argparser.add_argument("--max_epochs", default=400, type=int)
    # ------------------------------- SAM -------------------------------------
    argparser.add_argument("--sam_rho", type=float, default=0.1)
    # ------------------------------- SWA -------------------------------------
    argparser.add_argument("--is_swa", type=str2bool, default=True)
    argparser.add_argument("--swa_starts", nargs="+", default=[0.5, 0.6, 0.75, 0.9])
    argparser.add_argument("--swa_lr", type=float, default=0.05)
    argparser.add_argument("--swa_scheduler", type=str2bool, default=False)
    return argparser


NUM_CLASSES = {"svhn": 10, "cifar10": 10, "cifar100": 100, "imagenet": 1000}


def add_num_classes(experiment_args: dict):
    experiment_args["num_classes"] = NUM_CLASSES[experiment_args["dataset_name"]]


def parse_config() -> TrainConfig:
    parser = get_parser()
    experiment_args = vars(parser.parse_args())
    add_num_classes(experiment_args)
    return TrainConfig(**experiment_args)
