from dataclasses import dataclass




@dataclass
class TrainConfig:
    max_steps : int = 35_000
    eval_interval: int = 500
    eval_iters : int = 25
    stats_iter : int = 100
    global_batch_size: int = 256
    micro_batch_size: int = 16
    max_lr : int = 6e-4
    min_lr: int = 6e-5
    warmup_steps : int = max_steps // 20 # 5% of max_steps
