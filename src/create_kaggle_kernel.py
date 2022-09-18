from argparse import ArgumentParser
from pathlib import Path

from args import prepare_args, prepare_parser
from config import load_config
from const import COMPETITION_ABBREVIATION
from utils.kaggle import create_kaggle_kernel


def _add_args(parent_parser: ArgumentParser) -> ArgumentParser:
    parser = parent_parser.add_argument_group("create_kaggle_kernel_args")
    parser.add_argument("--training_start_timestamp", type=str)
    return parent_parser


if __name__ == "__main__":
    args = prepare_args(_add_args(prepare_parser()))
    config = load_config(args.config_path)
    _id = f"{COMPETITION_ABBREVIATION}-{args.training_start_timestamp}"
    create_kaggle_kernel(
        _id,
        "main.ipynb",
        dataset_sources=[f"shutotakahashi/{_id}"],
        save_dir=Path(config.base.output_dir)
        / "kaggle_kernels"
        / args.training_start_timestamp,
    )
