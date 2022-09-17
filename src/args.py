from argparse import ArgumentParser, Namespace


def add_system_level_args(parent_parser: ArgumentParser) -> ArgumentParser:
    parser = parent_parser.add_argument_group("system_level")
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--use_kaggle_secret", action="store_true")
    parser.add_argument("--wandb_api_key", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    return parent_parser


def prepare_parser() -> ArgumentParser:
    parser = ArgumentParser()
    return parser


def prepare_args(parser: ArgumentParser) -> Namespace:
    parser = add_system_level_args(parser)
    return parser.parse_args()
