from argparse import ArgumentParser, Namespace

from const import FILENAME


def add_system_level_args(parent_parser: ArgumentParser) -> ArgumentParser:
    parser = parent_parser.add_argument_group("system_level")
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--ensemble_config_path", type=str, default=None)
    parser.add_argument("--create_kaggle_dataset", action="store_true")
    parser.add_argument(
        "--map_hugging_face_model_name_to_kaggle_dataset", action="store_true"
    )
    parser.add_argument("--use_kaggle_secret", action="store_true")
    parser.add_argument("--wandb_api_key", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    return parent_parser


def add_val_args(parent_parser: ArgumentParser) -> ArgumentParser:
    parser = parent_parser.add_argument_group("val")
    parser.add_argument("--model_saved_dir", type=str, default=None)
    parser.add_argument("--checkpoint_filename", type=str, default=FILENAME.CHECKPOINT)
    parser.add_argument("--oof_save_dir", type=str, default=".")
    return parent_parser


def prepare_parser() -> ArgumentParser:
    parser = ArgumentParser()
    return parser


def prepare_args(parser: ArgumentParser) -> Namespace:
    parser = add_system_level_args(parser)
    parser = add_val_args(parser)
    return parser.parse_args()
