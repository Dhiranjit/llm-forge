from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class DataConfig:
    data_dir: str
    seq_len: int
    datasets: dict[str, float]


class _ShardReader:
    """Sequential memory-mapped token shard reader for a single dataset split."""

    def __init__(self, split_dir: Path, seq_len: int):
        self.seq_len = seq_len
        self.shards = sorted(split_dir.glob("*.bin"))
        assert len(self.shards) > 0, f"No shards found in {split_dir}."
        self.reset()

    def reset(self):
        self.shard_idx = 0
        self.pos = 0
        self.data = self._load_shard(self.shard_idx)

    def _load_shard(self, shard_idx: int) -> np.memmap:
        return np.memmap(self.shards[shard_idx], dtype=np.uint16, mode="r")

    def state_dict(self) -> dict:
        return {"shard_idx": self.shard_idx, "pos": self.pos}

    def load_state_dict(self, state: dict):
        self.shard_idx = state["shard_idx"]
        self.pos = state["pos"]
        assert 0 <= self.shard_idx < len(self.shards), (
            f"Invalid shard index {self.shard_idx} for {len(self.shards)} shards."
        )
        self.data = self._load_shard(self.shard_idx)

    def next_sequences(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        T = self.seq_len
        if self.pos + n * T + 1 > len(self.data):
            self.shard_idx = (self.shard_idx + 1) % len(self.shards)
            self.data = self._load_shard(self.shard_idx)
            self.pos = 0

        buff = self.data[self.pos : self.pos + n * T + 1]
        self.pos += n * T
        return buff[:-1].reshape(n, T), buff[1:].reshape(n, T)


class DataLoader:
    """Resumable multi-dataset token DataLoader with ratio-based within-batch mixing."""

    def __init__(
        self,
        cfg: DataConfig,
        split: str,
        batch_size: int,
        device: torch.device | str,
        seed: int,
    ):
        assert split in ("train", "val"), f"Invalid split: {split}"
        assert len(cfg.datasets) > 0, "data.datasets must not be empty"

        self.cfg = cfg
        self.batch_size = batch_size
        self.device = device
        self.seed = seed

        self.names = list(cfg.datasets.keys())
        weights = np.array([cfg.datasets[name] for name in self.names], dtype=np.float64)
        self.probs = weights / weights.sum()

        self.readers = {
            name: _ShardReader(Path(cfg.data_dir) / name / split, cfg.seq_len)
            for name in self.names
        }
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.seed)
        for reader in self.readers.values():
            reader.reset()

    def state_dict(self) -> dict:
        return {
            "readers": {name: r.state_dict() for name, r in self.readers.items()},
            "rng": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict):
        self.rng.bit_generator.state = state["rng"]
        for name, reader in self.readers.items():
            reader.load_state_dict(state["readers"][name])

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        if len(self.names) == 1:
            x_np, y_np = self.readers[self.names[0]].next_sequences(self.batch_size)
        else:
            counts = self.rng.multinomial(self.batch_size, self.probs)
            xs, ys = [], []
            for name, count in zip(self.names, counts):
                if count > 0:
                    x_i, y_i = self.readers[name].next_sequences(int(count))
                    xs.append(x_i)
                    ys.append(y_i)
            x_np = np.concatenate(xs, axis=0)
            y_np = np.concatenate(ys, axis=0)

        x = torch.from_numpy(x_np.astype(np.int64))
        y = torch.from_numpy(y_np.astype(np.int64))
        if str(self.device).startswith("cuda"):
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
        return x, y
