from dataclasses import dataclass


@dataclass
class TrainConfig:
    max_steps: int = 35_000
    eval_interval: int = 500
    eval_iters: int = 25
    stats_iter: int = 100
    global_batch_size: int = 256
    micro_batch_size: int = 16
    max_lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int | None = None

    def __post_init__(self) -> None:
        self.max_lr = float(self.max_lr)
        self.min_lr = float(self.min_lr)
        if self.warmup_steps is None:
            self.warmup_steps = self.max_steps // 20