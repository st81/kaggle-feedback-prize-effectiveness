import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel
from torch.utils.checkpoint import checkpoint

from config import Config
from utils.kaggle import HUGGING_FACE_MODEL_NAME_TO_KAGGLE_DATASET


def batch_padding(batch, is_test: bool = False):
    idx = int(torch.where(batch["attention_mask"] == 1)[1].max())

    idx += 1
    batch["attention_mask"] = batch["attention_mask"][:, :idx]
    batch["input_ids"] = batch["input_ids"][:, :idx]
    if not is_test:
        batch["target"] = batch["target"][:, :idx]
    batch["word_start_mask"] = batch["word_start_mask"][:, :idx]
    batch["word_ids"] = batch["word_ids"][:, :idx]

    return batch


class Net(nn.Module):
    def __init__(
        self,
        config: Config,
        is_test: bool = False,
        map_hugging_face_model_name_to_kaggle_dataset: bool = False,
    ) -> None:
        super(Net, self).__init__()

        self.config = config

        backbone = (
            HUGGING_FACE_MODEL_NAME_TO_KAGGLE_DATASET[config.architecture.backbone]
            if map_hugging_face_model_name_to_kaggle_dataset
            else config.architecture.backbone
        )
        hf_config = AutoConfig.from_pretrained(
            backbone,
            cache_dir=backbone
            if map_hugging_face_model_name_to_kaggle_dataset
            else None,
        )
        self.backbone = (
            AutoModel.from_pretrained(
                self.config.architecture.backbone, config=hf_config
            )
            if not is_test
            else AutoModel.from_config(hf_config)
        )

        if not is_test and self.config.architecture.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        self.head = nn.Linear(
            self.backbone.config.hidden_size, self.config.dataset.num_classes
        )
        self.loss_fn = nn.CrossEntropyLoss()
        if self.config.architecture.add_wide_dropout:
            raise NotImplementedError

    def forward(self, batch, calculate_loss=True, is_test: bool = False):
        outputs = {}

        if calculate_loss:
            outputs["target"] = batch["target"]
            outputs["word_start_mask"] = batch["word_start_mask"]

        batch = batch_padding(batch, is_test)
        if is_test:
            outputs["word_start_mask"] = batch["word_start_mask"]

        x = self.backbone(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ).last_hidden_state

        for obs_id in range(x.size()[0]):
            for w_id in range(int(torch.max(batch["word_ids"][obs_id]).item()) + 1):
                chunk_mask = batch["word_ids"][obs_id] == w_id
                chunk_logits = x[obs_id] * chunk_mask.unsqueeze(-1)
                chunk_logits = chunk_logits.sum(dim=0) / chunk_mask.sum()
                x[obs_id][chunk_mask] = chunk_logits

        if self.config.architecture.add_wide_dropout:
            raise NotImplementedError
        else:
            if not is_test and self.config.architecture.dropout > 0.0:
                x = F.dropout(
                    x, p=self.config.architecture.dropout, training=self.training
                )
            logits = self.head(x)

        if not self.training:
            if not is_test:
                outputs["logits"] = F.pad(
                    logits,
                    (0, 0, 0, self.config.tokenizer.max_length - logits.size()[1]),
                    "constant",
                    0,
                )
            else:
                outputs["logits"] = logits

        if calculate_loss:
            targets = batch["target"]
            new_logits = logits.view(-1, self.config.dataset.num_classes)

            if self.config.training.is_pseudo:
                raise NotImplementedError
            else:
                new_targets = targets.reshape(-1)
            new_word_start_mask = batch["word_start_mask"].reshape(-1)

            loss = self.loss_fn(new_logits, new_targets)
            outputs["loss"] = (
                loss * new_word_start_mask
            ).sum() / new_word_start_mask.sum()

        return outputs
