import collections
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from config import Config
from utils.types import PATH


def load_train_df(path: PATH) -> pd.DataFrame:
    if Path(path).suffix == ".pq":
        train_df = pd.read_parquet(path)
    else:
        train_df = pd.read_csv(path)
    return train_df


def split_train_val(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "fold" in train_df.columns:
        raise NotImplementedError()
    else:
        val_df = train_df.copy()
    return train_df, val_df


def change_df_for_debug(train_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame([train_df.loc[0, :].values for _ in range(len(train_df))])
    df.columns = train_df.columns
    return df


class CustomDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        config: Config,
        mode: str,
        label_encoder: Optional[LabelEncoder] = None,
    ) -> None:
        self.df = df.copy()
        self.config = config
        self.mode = mode
        self.label_encoder = label_encoder

        if mode == "train" and not config.training.is_pseudo:
            self.label_encoder = LabelEncoder().fit(
                np.concatenate(self.df[config.dataset.label_columns].values)
            )

        if not config.training.is_pseudo:
            self.df[config.dataset.label_columns] = self.df[
                config.dataset.label_columns
            ].map(lambda labels: self.label_encoder.transform(labels))

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.architecture.backbone, add_prefix_space=True, use_fast=True
        )
        self.text = self.df[self.config.dataset.text_column].values
        self.labels = self.df[self.config.dataset.label_columns].values

    def _read_data(self, idx: int, sample: Dict[str, Any]) -> Dict[str, Any]:
        text = self.text[idx]
        if (
            self.config.training.add_types is not None
            and self.config.training.add_types
        ):
            raise NotImplementedError()
        else:
            tokenizer_input = [list(text)]
        encodings = self.tokenizer(
            tokenizer_input,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.config.tokenizer.max_length,
            is_split_into_words=True,
        )

        sample["input_ids"] = encodings["input_ids"][0]
        sample["attention_mask"] = encodings["attention_mask"][0]

        word_ids = encodings.word_ids(0)
        word_ids = [-1 if x is None else x for x in word_ids]
        sample["word_ids"] = torch.tensor(word_ids)

        word_start_mask = []
        lab_idx = -1

        for i, word in enumerate(word_ids):
            word_start = word > -1 and (i == 0 or word_ids[i - 1] != word)
            if word_start:
                lab_idx += 1
                if self.config.training.is_pseudo:
                    if self.labels[idx][lab_idx][0] > 0:
                        word_start_mask.append(True)
                        continue
                else:
                    if self.labels[idx][lab_idx] != self.config.dataset.num_classes:
                        word_start_mask.append(True)
                        continue

            word_start_mask.append(False)

        sample["word_start_mask"] = torch.tensor(word_start_mask)
        return sample

    def _read_label(self, idx: int, sample: Dict[str, Any]) -> Dict[str, Any]:
        if self.config.training.is_pseudo:
            raise NotImplementedError()
        else:
            target = torch.full_like(sample["input_ids"], -100)

        word_start_mask = sample["word_start_mask"]

        if self.config.training.is_pseudo:
            raise NotImplementedError
        else:
            target[word_start_mask] = torch.tensor(
                [x for x in self.labels[idx] if x != self.config.dataset.num_classes]
            )
        sample["target"] = target
        return sample

    def __getitem__(self, index: int):
        sample = dict()
        sample = self._read_data(index, sample)
        sample = self._read_label(index, sample)
        return sample

    def __len__(self):
        return len(self.df)

    @staticmethod
    def get_train_collate_fn(config: Config):
        return None

    @staticmethod
    def get_val_collate_fn(config: Config):
        return None

    @staticmethod
    def batch_to_device(batch, device):
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        elif isinstance(batch, collections.abc.Mapping):
            return {
                key: CustomDataset.batch_to_device(value, device)
                for key, value in batch.items()
            }
        else:
            raise ValueError(f"Can not move {type(batch)} to device.")
