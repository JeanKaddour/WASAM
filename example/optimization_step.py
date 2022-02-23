import torch

from example.config import TrainConfig
from sam import SAM, disable_running_stats, enable_running_stats


def get_gradient_norm(model: torch.nn.Module) -> float:
    """Computes gradient norm of model."""
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


def SAM_optimization_step(
    model: torch.nn.Module,
    optimizer: SAM,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    criterion: torch.nn.Module,
) -> dict:
    # first forward-backward pass (use this pass for logging)
    enable_running_stats(model)
    predictions = model.forward(inputs)
    loss = criterion(input=predictions, target=labels)
    loss.backward()
    optimizer.first_step()

    # second forward-backward pass (ignore this pass' statistics)
    disable_running_stats(model)
    criterion(input=model.forward(inputs), target=labels).backward()
    gradient_norm = get_gradient_norm(model=model)

    optimizer.second_step()
    step = {
        "loss": loss.item(),
        "grad_norm": gradient_norm,
        "predictions": predictions,
    }
    return step


def one_optimization_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    criterion: torch.nn.Module,
) -> dict:
    optimizer.zero_grad()
    predictions = model.forward(inputs)
    loss = criterion(input=predictions, target=labels)
    loss.backward()
    gradient_norm = get_gradient_norm(model=model)
    optimizer.step()
    step = {
        "loss": loss.item(),
        "grad_norm": gradient_norm,
        "predictions": predictions,
    }
    return step


def optimization_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    criterion: torch.nn.Module,
    config: TrainConfig,
) -> dict:
    if config.optimizer_name == "sam":
        return SAM_optimization_step(
            model=model,
            optimizer=optimizer,
            inputs=inputs,
            labels=labels,
            criterion=criterion,
        )
    else:
        return one_optimization_step(
            model=model,
            optimizer=optimizer,
            inputs=inputs,
            labels=labels,
            criterion=criterion,
        )
