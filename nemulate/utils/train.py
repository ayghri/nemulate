from torch.optim.lr_scheduler import LRScheduler


class WarmupScheduler(LRScheduler):
    def __init__(
        self, optimizer, warmup_steps, after_scheduler=None, last_epoch=-1
    ):
        self.warmup_steps = warmup_steps
        self.after_scheduler = after_scheduler
        self.finished_warmup = False
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_steps
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
