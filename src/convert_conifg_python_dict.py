from argparse import ArgumentParser
from dataclasses import asdict
import json
from typing import Any, Dict

from args import prepare_args, prepare_parser
from config import load_config


def _add_args(parent_parser: ArgumentParser) -> ArgumentParser:
    parser = parent_parser.add_argument_group("convert_args")
    parser.add_argument("--not_mutate_input_path", action="store_true")
    parser.add_argument("--not_mutate_num_workers", action="store_true")
    return parent_parser


def _mutate_config_values(
    config: Dict[str, Any], mutate_input_path: bool, mutate_num_workers: bool
) -> Dict[str, Any]:
    for k, v in config.items():
        if type(v) == dict:
            _mutate_config_values(v, mutate_input_path, mutate_num_workers)
        if mutate_input_path and type(v) == str and "input/" in v:
            mutated = f"../{v}"
            print(f"Mutated config '{k}' from '{v}' to '{mutated}'")
            config[k] = mutated
        elif mutate_num_workers and k == "num_workers":
            mutated = 2
            print(f"Mutated config '{k}' from '{v}' to '{mutated}'")
            config[k] = mutated
    return config


if __name__ == "__main__":
    args = prepare_args(_add_args(prepare_parser()))
    config = load_config(args.config_path)
    config = asdict(config)
    config = _mutate_config_values(
        config, not args.not_mutate_input_path, not args.not_mutate_num_workers
    )
    print(
        json.dumps(config, indent=4).replace("true", "True").replace("false", "False")
    )
