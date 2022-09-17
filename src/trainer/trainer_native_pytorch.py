import gc
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from sklearn.metrics import log_loss
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import wandb

from config import Config
from data import CustomDataset
from models.feedback_model import Net
from scheduler import get_scheduler
from utils.types import PATH


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class TrainerNativePytorch:
    def __init__(
        self,
        config: Config,
        train_dataset: CustomDataset,
        eval_dataset: Optional[CustomDataset] = None,
        model_init: Optional[Callable[[Config], Net]] = None,
        save_dir: Optional[PATH] = None,
    ) -> None:
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.model_init = model_init
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self):
        train_loader = DataLoader(
            self.train_dataset,
            sampler=None,
            shuffle=False,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.environment.num_workers,
            pin_memory=False,
            collate_fn=self.train_dataset.get_train_collate_fn(self.config),
            drop_last=self.config.training.drop_last_batch,
            worker_init_fn=worker_init_fn,
        )
        print(
            f"train: dataset {len(self.train_dataset)}, dataloader: {len(train_loader)}"
        )
        if self.eval_dataset is not None:
            eval_loader = DataLoader(
                self.eval_dataset,
                shuffle=False,
                batch_size=1,
                num_workers=self.config.environment.num_workers,
                pin_memory=True,
                collate_fn=self.eval_dataset.get_val_collate_fn(self.config),
                worker_init_fn=worker_init_fn,
            )
        print(f"eval: dataset {len(self.eval_dataset)}, dataloader: {len(eval_loader)}")

        model = self.model_init(self.config)
        model.to(self.device)

        total_steps = len(self.train_dataset)

        no_decay = ["bias", "LayerNorm.weight"]
        differential_layers = self.config.training.differential_learning_rate_layers
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [
                        param
                        for name, param in model.named_parameters()
                        if (not any(layer in name for layer in differential_layers))
                        and (not any(nd in name for nd in no_decay))
                    ],
                    "lr": self.config.training.learning_rate,
                    "weight_decay": self.config.training.weight_decay,
                },
                {
                    "params": [
                        param
                        for name, param in model.named_parameters()
                        if (not any(layer in name for layer in differential_layers))
                        and (any(nd in name for nd in no_decay))
                    ],
                    "lr": self.config.training.learning_rate,
                    "weight_decay": 0,
                },
                {
                    "params": [
                        param
                        for name, param in model.named_parameters()
                        if (any(layer in name for layer in differential_layers))
                        and (not any(nd in name for nd in no_decay))
                    ],
                    "lr": self.config.training.differential_learning_rate,
                    "weight_decay": self.config.training.weight_decay,
                },
                {
                    "params": [
                        param
                        for name, param in model.named_parameters()
                        if (any(layer in name for layer in differential_layers))
                        and (any(nd in name for nd in no_decay))
                    ],
                    "lr": self.config.training.differential_learning_rate,
                    "weight_decay": 0,
                },
            ],
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

        scheduler = get_scheduler(self.config, optimizer, total_steps)

        if self.config.environment.mixed_precision:
            scaler = GradScaler()

        curr_step = 0
        i = 0
        best_val_loss = np.inf
        optimizer.zero_grad()
        for epoch in range(self.config.training.epochs):
            # set_seed(cfg.environment.seed + epoch)

            print(f"Epoch: {epoch}")

            progress_bar = tqdm(range(len(train_loader)))
            tr_it = iter(train_loader)
            losses = []
            gc.collect()
            model.train()

            for itr in progress_bar:
                i += 1
                curr_step += self.config.training.batch_size
                batch = CustomDataset.batch_to_device(next(tr_it), self.device)

                if self.config.environment.mixed_precision:
                    with autocast():
                        output_dict = model(batch)
                else:
                    output_dict = model(batch)

                loss = output_dict["loss"]
                losses.append(loss.item())

                if self.config.training.grad_accumulation != 1:
                    loss /= self.config.training.grad_accumulation

                if self.config.environment.mixed_precision:
                    scaler.scale(loss).backward()
                    if i % self.config.training.grad_accumulation == 0:
                        if self.config.training.gradient_clip > 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), self.config.training.gradient_clip
                            )
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    loss.backward()
                    if i % self.config.training.grad_accumulation == 0:
                        if self.config.training.gradient_clip > 0:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), self.config.training.gradient_clip
                            )
                        optimizer.step()
                        optimizer.zero_grad()

                if scheduler is not None:
                    scheduler.step()

                if curr_step % self.config.training.batch_size == 0:
                    progress_bar.set_description(
                        f"lr: {np.round(optimizer.param_groups[0]['lr'], 7)}, loss: {np.mean(losses[-10:]):.4f}"
                    )
                    print(curr_step)
                    if "wandb" in self.config.environment.report_to:
                        wandb.log(
                            {
                                "lr": np.round(optimizer.param_groups[0]["lr"], 7),
                                "train_loss": np.mean(losses[-10:]),
                            }
                        )

            progress_bar = tqdm(range(len(eval_loader)))
            val_it = iter(eval_loader)

            model.eval()
            preds = []
            probabilities = []
            all_targets = []
            for itr in progress_bar:
                data = next(val_it)
                batch = CustomDataset.batch_to_device(data, self.device)

                if self.config.environment.mixed_precision:
                    with autocast():
                        output_dict = model(batch)
                else:
                    output_dict = model(batch)

                if self.config.dataset.dataset_class == "feedback_dataset":
                    for logits, word_start_mask, target in zip(
                        output_dict["logits"],
                        output_dict["word_start_mask"],
                        output_dict["target"],
                    ):
                        # logits.shape: (batch_size * max_length * num_classes)
                        # word_start_mask.shape: (batch_size * max_length)
                        # target.shape: (batch_size * max_length)
                        probs = (
                            torch.softmax(logits[word_start_mask], dim=1)
                            .detach()
                            .cpu()
                            .numpy()
                        )  # shape: (num_word_start_mask_true, num_classes)
                        targets = (
                            target[word_start_mask].detach().cpu().numpy()
                        )  # shape: (num_word_start_mask_true, )

                        for p, t in zip(probs, targets):
                            probabilities.append(p)
                            if self.config.training.is_pseudo:
                                raise NotImplementedError
                            else:
                                all_targets.append(t)
                else:
                    raise NotImplementedError

            if self.config.dataset.dataset_class == "feedback_dataset":
                metric = log_loss(
                    all_targets,
                    np.vstack(probabilities),
                    eps=1e-7,
                    labels=list(range(self.config.dataset.num_classes)),
                )
            else:
                raise NotImplementedError
            print(f"Validation metric: {metric}")
            if "wandb" in self.config.environment.report_to:
                wandb.log({"val_log_loss": metric})

            if self.config.training.epochs > 0:
                checkpoint = {"model": model.state_dict()}
                torch.save(checkpoint, self.save_dir / "checkpoint.pth")
