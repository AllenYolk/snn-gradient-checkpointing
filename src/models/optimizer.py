import torch
from torch.optim import Optimizer


class Lomo(Optimizer):
    """https://github.com/OpenLMLab/LOMO/blob/main/lomo_optim/lomo.py
    """

    def __init__(
        self,
        optimizer: Optimizer,
        clip_grad_norm=None,
        clip_grad_value=None,
    ):
        self.optimizer = optimizer
        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_value = clip_grad_value
        self.need_scan = ((clip_grad_norm is not None) or
                          (clip_grad_value is not None))

        if self.clip_grad_norm is not None and self.clip_grad_norm <= 0:
            raise ValueError("clip_grad_norm must be positive if specified.")

        self.grad_func = self.fuse_update()
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    p.register_hook(self.grad_func)

        defaults = {
            "clip_grad_norm": clip_grad_norm,
            "clip_grad_value": clip_grad_value,
        }

        super().__init__(self.optimizer.param_groups, defaults)

    def fuse_update(self):

        def func(x):
            with torch.no_grad():
                for group in self.optimizer.param_groups:
                    for p in group["params"]:
                        if p.requires_grad and p.grad is not None:
                            grad = p.grad
                            if self.clip_grad_value is not None:
                                grad.clamp_(
                                    min=-self.clip_grad_value,
                                    max=self.clip_grad_value
                                )
                            if self.clip_grad_norm is not None:
                                raise NotImplementedError(
                                    "clip_grad_norm has not been implemented."
                                )
                            p.grad = grad

            self.optimizer.step()

            with torch.no_grad():
                for group in self.optimizer.param_groups:
                    for p in group["params"]:
                        if p.requires_grad:
                            p.grad = None

            return x

        def func_no_scan(x):
            self.optimizer.step()

            with torch.no_grad():
                for group in self.optimizer.param_groups:
                    for p in group["params"]:
                        if p.requires_grad:
                            p.grad = None

            return x

        return func if self.need_scan else func_no_scan

    def step(self):
        # The last parameter is not ready when calling the hook function.
        # Manually call the update function!
        self.grad_func(0)
