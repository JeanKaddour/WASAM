from pathlib import Path
from typing import Optional

import torch
import wandb
from init import get_starting_time, init, optimizer_to
from torch.optim.swa_utils import update_bn
from tqdm import tqdm

from example.config import TrainConfig
from example.dataset import get_train_valid_test_loader
from example.init import (
    get_criterion,
    get_device,
    get_lr_scheduler,
    get_model,
    get_optimizer,
    init_wandb_results,
)
from example.optimization_step import optimization_step
from swa import MultipleSWAModels


class Trainer:
    def __init__(
        self,
        config: TrainConfig,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional = None,
    ):
        init(config=config)
        self.config: TrainConfig = config
        self.device = get_device(config=config)
        self.model = (
            get_model(config=config).to(self.device)
            if model is None
            else model.to(self.device)
        )

        self.optimizer = (
            get_optimizer(
                optimizer_name=config.optimizer_name, config=config, model=self.model
            )
            if optimizer is None
            else optimizer
        )
        self.scheduler = get_lr_scheduler(config=config, optimizer=self.optimizer)
        (
            self.train_loader,
            self.valid_loader,
            self.test_loader,
        ) = get_train_valid_test_loader(config=config)
        self.criterion = get_criterion(config=config)
        self.scheduler = (
            get_lr_scheduler(config=config, optimizer=self.optimizer)
            if scheduler is None
            else scheduler
        )
        self.start_time = get_starting_time()
        optimizer_to(optim=self.optimizer, device=self.device)
        self.swa_models = MultipleSWAModels(
            base_model=self.model,
            device=self.device,
            max_epochs=config.max_epochs,
            starts=config.swa_starts,
        )
        init_wandb_results(
            swa_start_times=self.swa_models.swa_start_times,
        )
        self.max_val_acc, self.max_val_acc_swa = 0.0, 0.0
        self.epoch = 1
        self.output_path = f"{config.output_path}{config.dataset_name}/{config.model}/{config.optimizer_name}/{self.start_time}/"
        Path(self.output_path).mkdir(parents=True, exist_ok=True)

    def train_and_test_model(
        self,
    ) -> None:
        """Training loop with SWA."""
        for epoch in range(1, self.config.max_epochs + 1):
            self.train_for_one_epoch()
            self.eval_models()
            self.epoch += 1

    def log_metrics(self, metrics) -> None:
        wandb.log(metrics)

    def eval_models(self) -> None:
        self.eval_model(model=self.model)
        self.eval_swa_models()

    def eval_swa_models(self) -> None:
        for model_dict in self.swa_models.models:
            swa_model, swa_start = model_dict["model"], model_dict["start"]
            if self.epoch >= swa_start:
                suffix = f"_swa_{swa_start}"
                update_bn(loader=self.train_loader, model=swa_model, device=self.device)
                self.eval_model(model=swa_model, suffix=suffix)

    def eval_model(self, model: torch.nn.Module, suffix: str = "") -> None:
        val_metrics = self.model_evaluation(
            model=model,
            loader=self.valid_loader,
        )
        log_metrics = {}
        if val_metrics["acc"] > wandb.run.summary[f"valid{suffix}_best_acc"]:
            test_metrics = self.model_evaluation(
                model=model,
                loader=self.test_loader,
            )
            self.save_ckpt(
                model=model,
                file_name=f"best_valid{suffix}.pth",
            )
            log_metrics |= {f"test{suffix}": test_metrics, "epoch": self.epoch}
            wandb.run.summary[f"valid{suffix}_best_acc"] = val_metrics["acc"]
            wandb.run.summary[f"test{suffix}_best_acc"] = test_metrics["acc"]
        log_metrics |= {f"valid{suffix}": val_metrics, "epoch": self.epoch}
        self.log_metrics(log_metrics)
        model.train()

    def train_for_one_epoch(
        self,
    ) -> None:
        """Trains for one epoch."""
        self.model.train()
        epoch_loss = epoch_acc = 0.0
        for batch_idx, (inputs, labels) in enumerate(
            tqdm(self.train_loader, desc=f"Epoch {self.epoch}", leave=False)
        ):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            step_dict = optimization_step(
                model=self.model,
                optimizer=self.optimizer,
                inputs=inputs,
                labels=labels,
                criterion=self.criterion,
                config=self.config,
            )
            predictions = step_dict["predictions"]
            loss = step_dict["loss"]
            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == labels
            epoch_loss += loss
            epoch_acc += correct.sum()
        epoch_loss /= len(self.train_loader.dataset)
        epoch_acc /= len(self.train_loader.dataset)
        self.log_metrics(
            {"train": {"acc": epoch_acc, "loss": epoch_loss}, "epoch": self.epoch}
        )
        print(
            f"Epoch {self.epoch}: Train loss: {epoch_loss:.6f}, Train acc: {epoch_acc:.6f}"
        )
        self.swa_models.update_parameters(base_model=self.model, epoch=self.epoch)
        self.scheduler.step()

    def model_evaluation(
        self,
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
    ) -> dict[str, float]:
        """Computes loss and acc for given dataloader."""
        model.eval()
        total_loss, total_acc = 0.0, 0.0
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(
                tqdm(loader, desc="Validation: ", leave=False)
            ):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                predictions = model.forward(inputs)
                total_loss += self.criterion(input=predictions, target=labels).item()
                total_acc += (torch.argmax(predictions, dim=1) == labels).sum().item()
        total_loss /= len(loader.dataset)
        total_acc /= len(loader.dataset)
        return {"loss": total_loss, "acc": total_acc}

    def save_ckpt(
        self,
        model: torch.nn.Module,
        file_name: str,
    ) -> None:
        model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
        state = {
            "epoch": self.epoch,
            "model_state_dict": model_cpu,
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(state, self.output_path + file_name)
