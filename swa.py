from typing import List

import torch
from torch.optim.swa_utils import AveragedModel


class MultipleSWAModels:
    def __init__(
        self,
        base_model: torch.nn.Module,
        device: torch.device,
        max_epochs: int,
        starts: List[float],
    ) -> None:
        self.models = [
            {
                "model": AveragedModel(model=base_model, device=device),
                "start": int(start_percentage * max_epochs),
            }
            for start_percentage in starts
        ]
        self.swa_start_times = [model_dict["start"] for model_dict in self.models]
        self.max_epochs = max_epochs

    def update_parameters(self, base_model: torch.nn.Module, epoch: int) -> None:
        for model_dict in self.models:
            model, start = model_dict["model"], model_dict["start"]
            if epoch >= start:
                model.update_parameters(base_model)
