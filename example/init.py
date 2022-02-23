import random
from datetime import datetime
from typing import Callable, List, Optional, Union

import numpy as np
import torch
import wandb
from torch import nn as nn
from torch.optim.swa_utils import AveragedModel

from example.config import TrainConfig
from example.label_smoothing import LabelSmoothingLoss
from example.models import MODELS, PyramidNet, WideResNet
from sam import SAM


def optimizer_to(optim, device):
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def get_starting_time() -> str:
    return "{:%Y_%m_%d_%H_%M_%S_%f}".format(datetime.now())


def set_seeds(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init(config: TrainConfig) -> None:
    project_name = f"WASAM"
    run_name = f"{config.dataset_name}-{config.model}-{config.max_epochs}-{config.optimizer_name}-{str(config.seed)}"
    wandb.init(project=project_name, name=run_name, config=config)
    print(f"Config: {config}")
    set_seeds(config.seed)


def log_results(accuracies: dict[str, float], global_step: int, suffix: str = ""):
    wandb.log(
        {
            "global_step": global_step,
            f"valid{suffix}": accuracies["valid"],
            f"test{suffix}": accuracies["test"],
        }
    )
    if accuracies["valid"] > wandb.run.summary[f"valid.best_valid{suffix}"]:
        wandb.run.summary[f"best_valid{suffix}"] = accuracies["valid"]
        wandb.run.summary[f"best_test{suffix}"] = accuracies["test"]
        wandb.run.summary[f"best_step{suffix}"] = global_step


def init_wandb_results(swa_start_times: List[int]) -> None:
    metrics = [
        f"valid",
        f"test",
    ]
    strings = metrics[:]
    for start_time in swa_start_times:
        for metric in metrics:
            strings.append(metric + f"_swa_{start_time}")
    for string in strings:
        wandb.run.summary[f"{string}_best_acc"] = 0.0


def get_device(config: TrainConfig) -> Optional[torch.device]:
    return (
        torch.device(f"cuda:{config.device}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )


def get_optimizer_callable(config: TrainConfig) -> Callable:
    str_to_opt = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "adamW": torch.optim.AdamW,
    }
    if config.base_optimizer_name in str_to_opt:
        return str_to_opt[config.base_optimizer_name]
    raise NotImplementedError(
        f"Base Optimizer {config.base_optimizer_name} not implemented"
    )


class NoneScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer

    def step(self) -> None:
        pass

    def get_last_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]


def get_lr_scheduler(
    config: TrainConfig, optimizer: torch.optim.Optimizer
) -> Union[
    torch.optim.lr_scheduler.ExponentialLR,
    torch.optim.lr_scheduler.CosineAnnealingLR,
    torch.optim.lr_scheduler.CyclicLR,
    torch.optim.lr_scheduler.StepLR,
    torch.optim.lr_scheduler.MultiStepLR,
    NoneScheduler,
]:
    # See https://github.com/davda54/sam/issues/28
    if config.optimizer_name == "sam":
        optimizer = optimizer.base_optimizer

    if config.lr_scheduler == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_gamma)
    elif config.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.max_epochs
        )
    elif config.lr_scheduler == "cycle":
        return torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            0,
            max_lr=config.learning_rate,
            step_size_up=20,
            cycle_momentum=False,
        )
    elif config.lr_scheduler == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, milestones=[91, 136, 182]
        )
    elif config.lr_scheduler == "step":

        return torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=int(0.3 * config.max_epochs)
        )
    elif config.lr_scheduler == "none":
        return NoneScheduler(optimizer=optimizer)
    raise NotImplementedError(f"Scheduler {config.lr_scheduler} not implemented")


def get_criterion(config: TrainConfig) -> torch.nn.Module:
    if config.loss == "cross_entropy":
        return torch.nn.CrossEntropyLoss()
    elif config.loss == "label_smoothing":
        return LabelSmoothingLoss(config.label_smoothing_factor)
    raise NotImplementedError(f"Loss {config.loss} not implemented")


def get_SWA_model(
    config: TrainConfig,
    model: torch.nn.Module,
    device: torch.device,
) -> Optional[AveragedModel]:
    swa_model, swa_lr = None, None
    if config.is_swa:
        swa_model = AveragedModel(model=model, device=device)
    return swa_model


def get_optimizer(
    optimizer_name: str, config: TrainConfig, model: torch.nn.Module
) -> torch.optim.Optimizer:
    if optimizer_name == "sam":
        return SAM(
            params=model.parameters(),
            base_optimizer=get_optimizer(
                optimizer_name=config.base_optimizer_name, config=config, model=model
            ),
            rho=config.sam_rho,
            lr=config.learning_rate,
            momentum=config.sgd_momentum,
            weight_decay=config.weight_decay,
        )
    elif optimizer_name == "adam":
        return torch.optim.Adam(
            params=model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta_1, config.adam_beta_2),
            weight_decay=config.weight_decay,
        )
    elif optimizer_name == "adamW":
        return torch.optim.AdamW(
            params=model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta_1, config.adam_beta_2),
            weight_decay=config.weight_decay,
        )
    elif optimizer_name == "sgd":
        return torch.optim.SGD(
            params=model.parameters(),
            lr=config.learning_rate,
            momentum=config.sgd_momentum,
            weight_decay=config.weight_decay,
            nesterov=config.sgd_nesterov,
        )
    raise NotImplementedError(f"Optimizer {optimizer_name} not implemented")


def get_model(config: TrainConfig) -> nn.Module:
    assert config.model in MODELS, f"Invalid model name: {config.model}."
    if config.model == "pyramid":
        model = PyramidNet(
            dataset=config.dataset_name,
            depth=110,
            alpha=270,
            num_classes=config.num_classes,
        )
    elif config.model == "wrn":
        model = WideResNet(
            depth=28,
            width_factor=10,
            dropout=config.dropout,
            in_channels=3,
            labels=config.num_classes,
        )
    else:
        model = MODELS[config.model](
            pretrained=config.pretrained, num_classes=config.num_classes
        )
    return model
