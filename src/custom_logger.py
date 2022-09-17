from dataclasses import asdict
from typing import Optional

import wandb

from config import Config


def login_wandb() -> None:
    from kaggle_secrets import UserSecretsClient

    secret_label = "WANDB_API_KEY"
    secret_value = UserSecretsClient().get_secret(secret_label)
    wandb.login(key=secret_value)


def init_wandb(
    training_start_timestamp: str,
    config: Config,
    use_kaggle_secret: Optional[bool] = False,
    wandb_api_key: Optional[str] = None,
    wandb_group: Optional[str] = None,
) -> None:
    if use_kaggle_secret:
        login_wandb()

    if wandb_api_key is not None:
        wandb.login(key=wandb_api_key)

    wandb.init(
        config=asdict(config),
        project="kaggle-feedback-prize-effectiveness",
        group=training_start_timestamp if wandb_group is None else wandb_group,
        name=training_start_timestamp,
    )
