from dataclasses import dataclass
from genericpath import isdir
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml

import config
from const import FILENAME
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
    aux_type: bool = False
    custom_intermediate_dropout: bool = False
    intermediate_dropout: float = 0.0
    model_class: str = "feedback_model"
    no_type: Optional[bool] = None
    pool: str = ""
    use_sep: Optional[bool] = None
    use_type: bool = False
    wide_dropout: float = 0.0


@dataclass
class Dataset:
    dataset_class: str
    fold: int
    label_columns: Union[str, List[str]]
    num_classes: int
    text_column: Union[str, List[str]]
    train_df_path: str
    add_group_types: bool = True
    group_discourse: bool = True
    separator: str = ""


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
    add_newline_token: bool = True


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
    aux_loss_function: str = "CrossEntropy"
    mask_probability: float = 0.0


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
    if not path.is_file():
        # If path is file, use first yaml in directory as config file.
        path = list(path.glob("*.yaml"))[0]
    for k, v in load_yaml(path).items():
        _config[k] = getattr(config, k.capitalize())(**v)
    return Config(**_config)


@dataclass
class EnsembleConfig:
    model_saved_dir: str
    checkpoint_filename: str = FILENAME.CHECKPOINT


def load_ensemble_configs(path: PATH) -> List[EnsembleConfig]:
    return [EnsembleConfig(**c) for c in load_yaml(path)["configs"]]
