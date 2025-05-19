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
        scaler=None
    ):
        self.optimizer = optimizer
        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_value = clip_grad_value
        self.scaler = scaler

        if self.clip_grad_norm is not None and self.clip_grad_norm <= 0:
            raise ValueError("clip_grad_norm must be positive if specified.")

        self.grad_func = self._make_hook()
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    p.register_hook(self.grad_func)

        defaults = {
            "clip_grad_norm": clip_grad_norm,
            "clip_grad_value": clip_grad_value,
        }

        super().__init__(self.optimizer.param_groups, defaults)

        self._found_inf_or_nan = False

    @staticmethod
    def _has_inf_or_nan(x):
        try:
            # if x is half, the .float() incurs an additional deep copy, but it's necessary if
            # Pytorch's .sum() creates a one-element tensor of the same type as x
            # (which is true for some recent version of pytorch).
            cpu_sum = float(x.float().sum())
            # More efficient version that can be used if .sum() returns a Python scalar
            # cpu_sum = float(x.sum())
        except RuntimeError as e:
            # We want to check if inst is actually an overflow exception.
            # RuntimeError could come from a different error.
            # If so, we still want the exception to propagate.
            if "value cannot be converted" not in e.args[0]:
                raise
            return True
        else:
            if cpu_sum in [float("inf"), -float("inf")] or cpu_sum != cpu_sum:
                return True
            return False

    def _make_hook(self):

        def hook(x):
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

        def hook_no_clip(x):
            self.optimizer.step()

            with torch.no_grad():
                for group in self.optimizer.param_groups:
                    for p in group["params"]:
                        if p.requires_grad:
                            p.grad = None

            return x

        def hook_amp(x):
            scale = self.scaler._scale.item()
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
                            if self._has_inf_or_nan(grad):
                                self._found_inf_or_nan = True
                            else:
                                grad.div_(scale)
                            p.grad = grad

            if not self._found_inf_or_nan:
                self.optimizer.step()

            with torch.no_grad():
                for group in self.optimizer.param_groups:
                    for p in group["params"]:
                        if p.requires_grad:
                            p.grad = None

            return x

        if hasattr(self, "scaler") and self.scaler is not None:
            return hook_amp
        elif self.clip_grad_norm is not None or self.clip_grad_value is not None:
            return hook
        else:
            return hook_no_clip

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # The last parameter is not ready when calling the hook function.
        # Manually call the update function!
        self.grad_func(0)
        # The update of all parameters is done.
        # Set _found_inf_or_nan to False.
        self._found_inf_or_nan = False
        return loss
