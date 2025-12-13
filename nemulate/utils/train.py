from torch.optim.lr_scheduler import LRScheduler


class WarmupScheduler(LRScheduler):
    def __init__(
        self, optimizer, warmup_steps, after_scheduler=None, last_epoch=-1
    ):
        self.warmup_steps = warmup_steps
        self.after_scheduler = after_scheduler
        self.finished_warmup = False
        # Initialize like a normal LRScheduler. We keep last_epoch=-1 so that
        # before any call to step(), we are at epoch -1 and can return lr=0.
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup. We want lr=0 at initialization (last_epoch=-1)
            # and then a linear ramp from 0 to base_lr over `warmup_steps`
            # calls to `step()`. PyTorch increments last_epoch inside
            # LRScheduler.step(), so epoch 0 corresponds to the first
            # scheduler.step() call.
            effective_epoch = max(self.last_epoch, 0)
            alpha = effective_epoch / float(self.warmup_steps)
            return [base_lr * alpha for base_lr in self.base_lrs]

        if self.after_scheduler:
            if not self.finished_warmup:
                self.after_scheduler.base_lrs = self.base_lrs
                self.finished_warmup = True
            return self.after_scheduler.get_lr()

        return self.base_lrs

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_steps and self.after_scheduler:
            self.after_scheduler.step()
        super().step(epoch)
