"""
Adopted from:
https://github.com/davda54/sam/blob/main/sam.py
https://github.com/pytorch/contrib/blob/master/torchcontrib/optim/swa.py
"""
import warnings
from typing import Any, Callable, Dict, Iterable, Optional, Union

import torch
from torch.nn.modules.batchnorm import _BatchNorm

Params = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]
State = Dict[str, Any]
LossClosure = Callable[[], float]
OptLossClosure = Optional[LossClosure]


class WASAM(torch.optim.Optimizer):
    def __init__(
        self,
        params: Params,
        base_optimizer: torch.optim.Optimizer,
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs,
    ):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(WASAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer

    @torch.no_grad()
    def first_step(self, zero_grad: bool = True) -> None:
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (
                    (torch.pow(p, 2) if group["adaptive"] else 1.0)
                    * p.grad
                    * scale.to(p)
                )
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = True) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure: OptLossClosure = None):
        assert (
            closure is not None
        ), "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(
            closure
        )  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step(zero_grad=True)

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0
        ].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad)
                    .norm(p=2)
                    .to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm

    def update_swa(self):
        r"""Updates the SWA running averages of all optimized parameters."""
        for group in self.param_groups:
            self.update_swa_group(group)

    def update_swa_group(self, group):
        r"""Updates the SWA running averages for the given parameter group.

        Arguments:
            param_group (dict): Specifies for what parameter group SWA running
                averages should be updated

        """
        for p in group["params"]:
            param_state = self.state[p]
            if "swa_buffer" not in param_state:
                param_state["swa_buffer"] = torch.zeros_like(p.data)
            buf = param_state["swa_buffer"]
            virtual_decay = 1 / float(group["n_avg"] + 1)
            diff = (p.data - buf) * virtual_decay
            buf.add_(diff)
        group["n_avg"] += 1

    def swap_swa_sgd(self):
        r"""Swaps the values of the optimized variables and swa buffers.

        It's meant to be called in the end of training to use the collected
        swa running averages. It can also be used to evaluate the running
        averages during training; to continue training `swap_swa_sgd`
        should be called again.
        """
        for group in self.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                if "swa_buffer" not in param_state:
                    # If swa wasn't applied we don't swap params
                    warnings.warn(
                        "SWA wasn't applied to param {}; skipping it".format(p)
                    )
                    continue
                buf = param_state["swa_buffer"]
                tmp = torch.empty_like(p.data)
                tmp.copy_(p.data)
                p.data.copy_(buf)
                buf.copy_(tmp)

    def state_dict(self):
        r"""Returns the state of SWA as a :class:`dict`.

        It contains three entries:
            * opt_state - a dict holding current optimization state of the base
                optimizer. Its content differs between optimizer classes.
            * swa_state - a dict containing current state of SWA. For each
                optimized variable it contains swa_buffer keeping the running
                average of the variable
            * param_groups - a dict containing all parameter groups
        """
        opt_state_dict = self.base_optimizer.state_dict()
        swa_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        opt_state = opt_state_dict["state"]
        param_groups = opt_state_dict["param_groups"]
        return {
            "opt_state": opt_state,
            "swa_state": swa_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        r"""Loads the optimizer state.

        Args:
            state_dict (dict): SWA optimizer state. Should be an object returned
                from a call to `state_dict`.
        """
        swa_state_dict = {
            "state": state_dict["swa_state"],
            "param_groups": state_dict["param_groups"],
        }
        opt_state_dict = {
            "state": state_dict["opt_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(WASAM, self).load_state_dict(swa_state_dict)
        self.base_optimizer.load_state_dict(opt_state_dict)
        self.opt_state = self.base_optimizer.state
        self.base_optimizer.param_groups = self.param_groups

    @staticmethod
    def bn_update(loader, model, device=None):
        r"""Updates BatchNorm running_mean, running_var buffers in the model.

        It performs one pass over data in `loader` to estimate the activation
        statistics for BatchNorm layers in the model.

        Args:
            loader (torch.utils.data.DataLoader): dataset loader to compute the
                activation statistics on. Each data batch should be either a
                tensor, or a list/tuple whose first element is a tensor
                containing data.

            model (torch.nn.Module): model for which we seek to update BatchNorm
                statistics.

            device (torch.device, optional): If set, data will be trasferred to
                :attr:`device` before being passed into :attr:`model`.
        """
        if not _check_bn(model):
            return
        was_training = model.training
        model.train()
        momenta = {}
        model.apply(_reset_bn)
        model.apply(lambda module: _get_momenta(module, momenta))
        n = 0
        for input in loader:
            if isinstance(input, (list, tuple)):
                input = input[0]
            b = input.size(0)

            momentum = b / float(n + b)
            for module in momenta.keys():
                module.momentum = momentum

            if device is not None:
                input = input.to(device)

            model(input)
            n += b

        model.apply(lambda module: _set_momenta(module, momenta))
        model.train(was_training)


# BatchNorm utils
def _check_bn_apply(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def _check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn_apply(module, flag))
    return flag[0]


def _reset_bn(module):
    if issubclass(module.__class__, _BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, _BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, _BatchNorm):
        module.momentum = momenta[module]


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)
