import collections
import gc
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import wandb

from config import Config
from const import FILENAME
from data import CustomDataset
from data_es import EssayDataset
from models.feedback_model import Net
from models.util import load_checkpoint
from scheduler import get_scheduler
from utils.kaggle import create_submission
from utils.seed import set_seed
from utils.types import PATH


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class TrainerNativePytorch:
    def __init__(
        self,
        config: Config,
        train_dataset: Optional[Union[CustomDataset, EssayDataset]] = None,
        eval_dataset: Optional[Union[CustomDataset, EssayDataset]] = None,
        model_init: Optional[Callable[[Config], Net]] = None,
        save_dir: Optional[PATH] = None,
        val_df: Optional[pd.DataFrame] = None,
        raw_val_df: Optional[pd.DataFrame] = None,
    ) -> None:
        if isinstance(eval_dataset, EssayDataset) and val_df is None:
            raise ValueError(
                "'val_df' must be provided if 'type(eval_dataset)' is 'EssayDataset'"
            )

        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.model_init = model_init
        if save_dir is not None:
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = None
        self.val_df = val_df
        self.raw_val_df = raw_val_df
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_loader = DataLoader(
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
            f"train: dataset {len(self.train_dataset)}, dataloader: {len(self.train_loader)}"
        )
        if self.eval_dataset is not None:
            self.eval_loader = DataLoader(
                self.eval_dataset,
                shuffle=False,
                batch_size=1,
                num_workers=self.config.environment.num_workers,
                pin_memory=True,
                collate_fn=self.eval_dataset.get_val_collate_fn(self.config),
                worker_init_fn=worker_init_fn,
            )
            print(
                f"eval: dataset {len(self.eval_dataset)}, dataloader: {len(self.eval_loader)}"
            )

    def train(self):
        model = self.model_init(self.config)
        model.to(self.device)

        if self.config.architecture.pretrained_weights != "":
            try:
                load_checkpoint(self.config, model)
            except Exception as e:
                print(e)
                print("WARNING: could not load checkpoint")

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
            set_seed(self.config.environment.seed + epoch)

            print(f"Epoch: {epoch}")

            progress_bar = tqdm(range(len(self.train_loader)))
            tr_it = iter(self.train_loader)
            losses = []
            gc.collect()
            model.train()

            for itr in progress_bar:
                i += 1
                curr_step += self.config.training.batch_size
                if self.config.dataset.dataset_class == "feedback_dataset":
                    batch = CustomDataset.batch_to_device(next(tr_it), self.device)
                elif self.config.dataset.dataset_class == "feedback_dataset_essay_ds":
                    batch = EssayDataset.batch_to_device(next(tr_it), self.device)

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

                del batch, output_dict, loss
                gc.collect()

            if self.config.training.epochs > 0:
                checkpoint = {"model": model.state_dict()}
                torch.save(checkpoint, self.save_dir / FILENAME.CHECKPOINT)

            progress_bar = tqdm(range(len(self.eval_loader)))
            val_it = iter(self.eval_loader)

            model.eval()
            preds = []
            probabilities = []
            all_targets = []

            with torch.no_grad():
                for itr in progress_bar:
                    data = next(val_it)
                    if self.config.dataset.dataset_class == "feedback_dataset":
                        batch = CustomDataset.batch_to_device(data, self.device)
                    elif (
                        self.config.dataset.dataset_class == "feedback_dataset_essay_ds"
                    ):
                        batch = EssayDataset.batch_to_device(data, self.device)

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
                                    all_targets.append(np.argmax(t))
                                else:
                                    all_targets.append(t)
                    else:
                        preds.append(
                            output_dict["logits"]
                            .float()
                            .softmax(dim=1)
                            .detach()
                            .cpu()
                            .numpy()
                        )

                if self.config.dataset.dataset_class == "feedback_dataset":
                    metric = log_loss(
                        all_targets,
                        np.vstack(probabilities),
                        eps=1e-7,
                        labels=list(range(self.config.dataset.num_classes)),
                    )
                else:
                    preds = np.concatenate(preds, axis=0)
                    metric = log_loss(
                        self.val_df[self.config.dataset.label_columns].values.argmax(
                            axis=1
                        ),
                        preds,
                    )
                print(f"Validation metric: {metric}")
                if "wandb" in self.config.environment.report_to:
                    wandb.log({"val_log_loss": metric})

    def validate(self, model_saved_path: PATH, oof_save_dir: PATH) -> None:
        if self.raw_val_df is None:
            print(
                "Warning: Validation results will not be saved because 'raw_val_df' is not provided"
            )
        Path(oof_save_dir).mkdir(parents=True, exist_ok=True)

        model = self.model_init(self.config)
        model.to(self.device).eval()

        d = torch.load(model_saved_path, map_location="cpu")
        model_weights = d["model"]
        model_weights = {k.replace("module.", ""): v for k, v in model_weights.items()}
        model.load_state_dict(collections.OrderedDict(model_weights), strict=True)
        del d, model_weights
        gc.collect()

        progress_bar = tqdm(range(len(self.eval_loader)))
        val_it = iter(self.eval_loader)

        preds = []
        probabilities = []
        all_targets = []

        with torch.no_grad():
            for itr in progress_bar:
                print(" ")
                data = next(val_it)
                if self.config.dataset.dataset_class == "feedback_dataset":
                    batch = CustomDataset.batch_to_device(data, self.device)
                elif self.config.dataset.dataset_class == "feedback_dataset_essay_ds":
                    batch = EssayDataset.batch_to_device(data, self.device)

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
                                all_targets.append(np.argmax(t))
                            else:
                                all_targets.append(t)
                else:
                    preds.append(
                        output_dict["logits"]
                        .float()
                        .softmax(dim=1)
                        .detach()
                        .cpu()
                        .numpy()
                    )

            if self.config.dataset.dataset_class == "feedback_dataset":
                metric = log_loss(
                    all_targets,
                    np.vstack(probabilities),
                    eps=1e-7,
                    labels=list(range(self.config.dataset.num_classes)),
                )
                if self.raw_val_df is not None:
                    df = create_submission(
                        self.raw_val_df["discourse_id"], np.vstack(probabilities)
                    )
                    df.to_csv(
                        Path(oof_save_dir)
                        / FILENAME.oof_filename(
                            self.config.dataset.fold, self.config.environment.seed
                        ),
                        index=False,
                    )
            else:
                preds = np.concatenate(preds, axis=0)
                metric = log_loss(
                    self.val_df[self.config.dataset.label_columns].values.argmax(
                        axis=1
                    ),
                    preds,
                )
            print(f"Validation metric: {metric}")
            if "wandb" in self.config.environment.report_to:
                wandb.log({"val_log_loss": metric})

    def predict(
        self, test_dataset: Union[CustomDataset, EssayDataset], model_saved_path: PATH
    ) -> np.ndarray:
        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=4 if isinstance(test_dataset, CustomDataset) else 8,
            num_workers=2,
        )

        model = self.model_init(self.config)
        model.to(self.device).eval()

        d = torch.load(model_saved_path, map_location="cpu")
        model_weights = d["model"]
        model_weights = {k.replace("module.", ""): v for k, v in model_weights.items()}
        model.load_state_dict(collections.OrderedDict(model_weights), strict=True)
        del d, model_weights
        gc.collect()

        preds = []
        with torch.no_grad():
            for data in tqdm(test_loader):
                print(" ")
                if self.config.dataset.dataset_class == "feedback_dataset":
                    batch = CustomDataset.batch_to_device(data, self.device)
                elif self.config.dataset.dataset_class == "feedback_dataset_essay_ds":
                    batch = EssayDataset.batch_to_device(data, self.device)

                if self.config.environment.mixed_precision:
                    with autocast():
                        output_dict = model(batch, calculate_loss=False, is_test=True)
                else:
                    output_dict = model(batch, calculate_loss=False, is_test=True)

                if self.config.dataset.dataset_class == "feedback_dataset":
                    # output_dict:
                    #   logits.shape: (batch_size * max_length in batch * num_classes)
                    #   word_start_mask.shape: (batch_size * max_length in batch)
                    #   target.shape: (batch_size * max_length in batch)
                    probs = (
                        torch.softmax(
                            output_dict["logits"][output_dict["word_start_mask"]], dim=1
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )  # shape: (num_word_start_mask_true, num_classes)
                    preds.append(probs)

                else:
                    # output_dict:
                    #   logits.shape: (num discourse , num_classes)
                    preds.append(
                        output_dict["logits"]
                        .float()
                        .softmax(dim=1)
                        .detach()
                        .cpu()
                        .numpy()
                    )

        return preds
