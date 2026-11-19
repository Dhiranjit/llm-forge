from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from torch import nn

from llm_forge.data.loader import DataConfig
from llm_forge.models.gpt2 import GPT2, GPTConfig
from llm_forge.trainer.config import TrainConfig

# Maps `model.arch` in YAML to its model config dataclass and nn.Module class.
# Add `"forge": ForgeConfig` / `"forge": Forge` here once defined.
MODEL_CONFIG_REGISTRY: dict[str, type[Any]] = {
    "gpt2": GPTConfig,
}

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "gpt2": GPT2,
}

type ModelConfig = GPTConfig


def build_model(arch: str, model_cfg: ModelConfig) -> nn.Module:
    """Instantiate a model from its architecture name and typed config."""
    if arch not in MODEL_REGISTRY:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model arch {arch!r}. Supported: {supported}")
    return MODEL_REGISTRY[arch](model_cfg)


@dataclass
class ExperimentConfig:
    arch: str
    model: ModelConfig
    data: DataConfig
    train: TrainConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": {"arch": self.arch, **asdict(self.model)},
            "data": asdict(self.data),
            "train": asdict(self.train),
        }


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment YAML file into a typed ExperimentConfig."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}, got {type(raw).__name__}")

    for section in ("model", "data", "train"):
        if section not in raw or not isinstance(raw[section], dict):
            raise ValueError(f"Missing or invalid '{section}' section in {config_path}")

    model_section = dict(raw["model"])
    arch = model_section.pop("arch", None)
    if arch not in MODEL_CONFIG_REGISTRY:
        supported = ", ".join(sorted(MODEL_CONFIG_REGISTRY))
        raise ValueError(f"Unknown model arch {arch!r}. Supported: {supported}")

    model_cfg_cls = MODEL_CONFIG_REGISTRY[arch]

    return ExperimentConfig(
        arch=arch,
        model=model_cfg_cls(**model_section),
        data=DataConfig(**raw["data"]),
        train=TrainConfig(**raw["train"]),
    )
