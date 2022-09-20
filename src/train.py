from pathlib import Path
import subprocess

from dotenv import load_dotenv

from args import prepare_args, prepare_parser
from config import Config, load_config
from const import COMPETITION_ABBREVIATION
from data import change_df_for_debug, load_train_df, split_train_val, CustomDataset
from models.base import get_model
from trainer.trainer_native_pytorch import TrainerNativePytorch
from utils.seed import set_seed
from utils.time import now
from utils.kaggle import create_kaggle_dataset

from custom_logger import init_wandb


def train(config: Config, save_dir: str) -> None:
    load_dotenv()
    set_seed(config.environment.seed)

    train_df, val_df = split_train_val(
        load_train_df(config.dataset.train_df_path), config.dataset.fold
    )
    if config.environment.debug:
        train_df, val_df = change_df_for_debug(train_df), change_df_for_debug(val_df)
        train_df = train_df.loc[:50, :]
        val_df = val_df.loc[:1, :]
    print(f"Example label: {train_df[config.dataset.label_columns].values[0]}")

    train_dataset = CustomDataset(train_df, config, "train")
    val_dataset = CustomDataset(val_df, config, "val", train_dataset.label_encoder)

    trainer = TrainerNativePytorch(
        config, train_dataset, val_dataset, get_model, save_dir
    )
    trainer.train()


if __name__ == "__main__":
    args = prepare_args(prepare_parser())
    config = load_config(args.config_path)
    # config = load_config("config/pretrain_2021.yaml")
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
