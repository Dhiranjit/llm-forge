from dataclasses import dataclass


@dataclass
class TrainConfig:
    max_steps: int
    batch_size: int
    lr: float
    eval_interval: int
    eval_iters: int
    log_interval: int
    grad_accum_steps: int = 1
    warmup_steps: int = 0
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 1337
