import numpy as np


class ScheduledOptim:
    def __init__(self, optimizer, init_lr, n_warmup_epoch=10, update_rate=5, decay_rate=0.9):
        self._optimizer = optimizer
        self.n_warmup_epoch = n_warmup_epoch
        self.n_current_steps = 0
        self.init_lr = init_lr
        self.update_rate = update_rate
        self.decay_rate = decay_rate

    def step_and_update_lr(self):
        self.n_current_steps += 1
        rate = self._update_learning_rate()
        return rate

    def zero_grad(self):
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        # decay_rate的(current_steps - warmup_epoch + 1 // update_rate + 1) 次方
        # 也就是过了warmup_epoch之后每过update_rate个epoch，learning_rate乘以decay_rate
        return np.power(self.decay_rate, max((self.n_current_steps - self.n_warmup_epoch + 1) // self.update_rate + 1, 0))

    def _update_learning_rate(self):

        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
        return lr
