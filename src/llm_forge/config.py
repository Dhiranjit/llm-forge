from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from torch import nn

from llm_forge.data.loader import DataConfig
from llm_forge.models.gpt2 import GPT2, GPTConfig
from llm_forge.trainer.config import TrainConfig

MODELS: dict[str, tuple[type[nn.Module], type[Any]]] = {
    "gpt2": (GPT2, GPTConfig),
}

type ModelConfig = GPTConfig


@dataclass
class ExperimentConfig:
    arch: str
    model: ModelConfig
    data: DataConfig
    train: TrainConfig

    def build_model(self) -> nn.Module:
        model_cls, _ = MODELS[self.arch]
        return model_cls(self.model)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": {"arch": self.arch, **asdict(self.model)},
            "data": asdict(self.data),
            "train": asdict(self.train),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        model_section = dict(raw["model"])
        arch = model_section.pop("arch")
        _, model_cfg_cls = MODELS[arch]
        return cls(
            arch=arch,
            model=model_cfg_cls(**model_section),
            data=DataConfig(**raw["data"]),
            train=TrainConfig(**raw["train"]),
        )


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment YAML file into a typed ExperimentConfig."""
    with Path(path).open("r", encoding="utf-8") as f:
        return ExperimentConfig.from_dict(yaml.safe_load(f))
