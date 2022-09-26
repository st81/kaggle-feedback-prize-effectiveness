from functools import partial
from pathlib import Path
import subprocess

from dotenv import load_dotenv
import numpy as np

from args import prepare_args, prepare_parser
from config import Config, load_config
from const import COMPETITION_ABBREVIATION
from data import change_df_for_debug, load_train_df, split_train_val, CustomDataset
from data_es import EssayDataset
from models.base import get_model
from trainer.trainer_native_pytorch import TrainerNativePytorch
from utils.env import set_environ
from utils.seed import set_seed
from utils.time import now
from utils.kaggle import create_kaggle_dataset

from custom_logger import init_wandb


def train(config: Config, save_dir: str) -> None:
    load_dotenv()
    if config.environment.seed < 0:
        config.environment.seed = np.random.randint(1_000_000)
    else:
        config.environment.seed = config.environment.seed
    print(f"seed: {config.environment.seed}")
    set_seed(config.environment.seed)

    train_df, val_df = split_train_val(
        load_train_df(config.dataset.train_df_path), config.dataset.fold
    )
    if config.environment.debug:
        train_df, val_df = change_df_for_debug(
            train_df, val_df, config, config.training.is_pseudo
        )
    print(train_df, val_df, sep="\n")
    print(f"Example label: {train_df[config.dataset.label_columns].values[0]}")

    if config.dataset.dataset_class == "feedback_dataset":
        train_dataset = CustomDataset(train_df, config, "train")
        val_dataset = CustomDataset(val_df, config, "val", train_dataset.label_encoder)
    elif config.dataset.dataset_class == "feedback_dataset_essay_ds":
        train_dataset = EssayDataset(train_df, config, "train")
        print("*" * 20)
        val_dataset = EssayDataset(val_df, config, "val")
    else:
        raise ValueError(
            "'config.dataset.dataset_class' must be either 'feedback_dataset' or 'feedback_dataset_essay_ds'."
        )

    trainer = TrainerNativePytorch(
        config,
        train_dataset,
        val_dataset,
        partial(get_model, model_class=config.architecture.model_class),
        save_dir,
        val_df=val_df,
    )
    trainer.train()


if __name__ == "__main__":
    set_environ()
    args = prepare_args(prepare_parser())
    config = load_config(args.config_path)
    # config = load_config("config/cfg_pseudo_75.yaml")
    training_start_timestamp = now()
    if "wandb" in config.environment.report_to:
        init_wandb(
            training_start_timestamp,
            config,
            args.use_kaggle_secret,
            wandb_api_key=args.wandb_api_key,
            wandb_group=args.wandb_group,
        )
    save_dir = Path(config.base.output_dir) / training_start_timestamp
    train(config, save_dir)

    subprocess.run(f"cp -r {args.config_path} {save_dir}", shell=True)
    if args.create_kaggle_dataset:
        create_kaggle_dataset(
            f"{COMPETITION_ABBREVIATION}-{training_start_timestamp}", save_dir
        )
