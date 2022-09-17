from torch.optim import Optimizer
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from config import Config


def get_scheduler(config: Config, optimizer: Optimizer, total_steps: int):
    args = {
        "optimizer": optimizer,
        "num_warmup_steps": int(
            config.training.warmup_epochs * (total_steps // config.training.batch_size)
        ),
        "num_training_steps": config.training.epochs
        * (total_steps // config.training.batch_size),
    }

    scheduler = (
        get_linear_schedule_with_warmup(**args)
        if config.training.schedule == "Linear"
        else get_cosine_schedule_with_warmup(**args)
    )
    return scheduler
