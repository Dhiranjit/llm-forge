import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass


@dataclass
class DataConfig:
    data_dir: str
    seq_len: int


class DataLoader:
    """Sequential memory-mapped token shard loader with resumable state."""
    def __init__(
        self,
        cfg: DataConfig,
        split: str,
        batch_size: int,
        device: torch.device | str,
    ):
        self.cfg = cfg
        self.batch_size = batch_size
        self.device = device

        # Shard discovery
        assert split in ("train", "val"), f"Invalid split: {split}"
        split_dir = Path(cfg.data_dir) / split
        self.shards = sorted(split_dir.glob("*.bin"))
        assert len(self.shards) > 0, f"No shards for '{split}' found in {split_dir}."

        self.reset()


    def reset(self):
        self.current_shard_idx = 0
        self.current_position = 0
        self.data = self._load_shard(self.current_shard_idx)

        
    def _load_shard(self, shard_idx):
        shard_path = self.shards[shard_idx]
        return np.memmap(shard_path, dtype=np.uint16, mode="r")


    def state_dict(self):
        return {
            "current_shard_idx": self.current_shard_idx,
            "current_position": self.current_position,
        }


    def load_state_dict(self, state):
        self.current_shard_idx = state["current_shard_idx"]
        self.current_position = state["current_position"]

        assert 0 <= self.current_shard_idx < len(self.shards), (
            f"Invalid shard index {self.current_shard_idx} for {len(self.shards)} shards."
        )
        self.data = self._load_shard(self.current_shard_idx)

        # Advance if remaining tokens can't fit a batch (e.g. larger B or T on resume)
        if self.current_position + (self.batch_size * self.cfg.seq_len + 1) > len(self.data):
            self.current_shard_idx = (self.current_shard_idx + 1) % len(self.shards)
            self.data = self._load_shard(self.current_shard_idx)
            self.current_position = 0

    def next_batch(self):
        B, T = self.batch_size, self.cfg.seq_len

        buff = self.data[self.current_position : self.current_position + B * T + 1]

        x_tensor = torch.from_numpy(buff[:-1].astype(np.int64)).view(B, T)
        y_tensor = torch.from_numpy(buff[1:].astype(np.int64)).view(B, T)

        x = x_tensor.pin_memory().to(self.device, non_blocking=True)
        y = y_tensor.pin_memory().to(self.device, non_blocking=True)

        self.current_position += B * T

        # Advance to the next shard if remaining tokens can't fit the next batch
        if self.current_position + (B * T + 1) > len(self.data):
            self.current_shard_idx = (self.current_shard_idx + 1) % len(self.shards)
            self.data = self._load_shard(self.current_shard_idx)
            self.current_position = 0

        return x, y
