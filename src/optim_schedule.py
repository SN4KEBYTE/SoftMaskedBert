import numpy as np
from torch.optim import Optimizer


class ScheduledOptim:
    def __init__(
        self,
        optimizer: Optimizer,
        d_model: int,
        n_warmup_steps: int,
    ) -> None:
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps

        self.n_current_steps = 0
        self.init_lr = 1 / np.sqrt(d_model)

    def step_and_update_lr(
        self,
    ) -> None:
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(
        self,
    ) -> None:
        self._optimizer.zero_grad()

    def _get_lr_scale(
        self,
    ) -> float:
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps,
        ])

    def _update_learning_rate(
        self,
    ) -> None:
        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
