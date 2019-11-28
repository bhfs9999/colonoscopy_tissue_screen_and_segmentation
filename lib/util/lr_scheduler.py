import torch
from bisect import bisect_right, bisect_left


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,
            gamma=0.1,
            warmup_factor=1.0 / 3,
            warmup_iters=500,
            warmup_method="linear",
            last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class CycleScheduler:
    def __init__(
            self,
            optimizer,
            lr_cycle,
            steps,
    ):
        self.optimizer = optimizer
        self.current_lr = optimizer.param_groups[0]['lr']
        self.lr_cycle = lr_cycle
        self.steps = steps
        self.direction = -1
        self.lrs = iter(self.lr_iter(self.lr_cycle, self.steps, self.current_lr))

    def lr_iter(self, lr_cycle, steps, lr):
        current_lr = lr
        lr_cycle = sorted(lr_cycle, reverse=True)
        intervals = []
        for i in range(len(lr_cycle) - 1):
            intervals.append((max(lr_cycle[i], lr_cycle[i + 1]) - min(lr_cycle[i], lr_cycle[i + 1])) / steps[i])

        while True:
            if current_lr >= max(lr_cycle) or current_lr <= min(lr_cycle):
                self.direction *= -1
            for i in range(len(lr_cycle) - 1):
                if lr_cycle[i] >= current_lr >= lr_cycle[i + 1]:
                    interval = intervals[i]
                    current_lr += self.direction * interval
                    current_lr = min(current_lr, max(lr_cycle))
                    current_lr = max(current_lr, min(lr_cycle))
            yield current_lr

    def get_lr(self):
        return self.current_lr

    def step(self):
        self.current_lr = next(self.lrs)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer' and key != 'lrs'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)
        self.lrs = iter(self.lr_iter(self.lr_cycle, self.steps, self.current_lr))
