from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

import config
from utils.types import PATH


@dataclass
class Base:
    feedback_prize_effectiveness_dir: str
    feedback_prize_2021_dir: str
    input_data_dir: str
    output_dir: str


@dataclass
class Architecture:
    backbone: str
    dropout: float
    gradient_checkpointing: bool
    add_wide_dropout: Optional[bool] = None
    pretrained_weights: Optional[str] = ""


@dataclass
class Dataset:
    dataset_class: str
    fold: int
    label_columns: str
    num_classes: int
    text_column: str
    train_df_path: str


@dataclass
class Environment:
    mixed_precision: bool
    num_workers: int
    report_to: List[str]
    seed: int
    debug: bool = False


@dataclass
class Tokenizer:
    max_length: int


@dataclass
class Training:
    batch_size: int
    differential_learning_rate: float
    differential_learning_rate_layers: List[str]
    drop_last_batch: bool
    epochs: int
    grad_accumulation: int
    gradient_clip: float
    is_pseudo: bool
    learning_rate: float
    loss_function: str
    optimizer: str
    schedule: str
    warmup_epochs: float
    weight_decay: float
    add_types: Optional[bool] = None


@dataclass
class Config:
    base: Base
    architecture: Architecture
    dataset: Dataset
    environment: Environment
    tokenizer: Tokenizer
    training: Training


def load_yaml(path: PATH) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.load(f, yaml.FullLoader)


def load_config(path: PATH) -> Config:
    path = Path(path)
    _config = {}
    for k, v in load_yaml(path).items():
        _config[k] = getattr(config, k.capitalize())(**v)
    return Config(**_config)
