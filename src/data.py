import collections
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from config import Config
from const import LABEL_O_WHEN_TEST
from utils.kaggle import HUGGING_FACE_MODEL_NAME_TO_KAGGLE_DATASET
from utils.types import PATH


def add_essay_text_column(
    df: pd.DataFrame, data_dir: PATH, is_test: bool = False
) -> pd.DataFrame:
    essay_texts = {}
    for filename in Path(data_dir).glob(f"{'test' if is_test else 'train'}/*.txt"):
        with open(filename, "r") as f:
            lines = f.read()
        essay_texts[filename.stem] = lines
    df["essay_text"] = df["essay_id"].map(essay_texts)
    return df


def modify_discourse_text(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[
        df.discourse_id == "56744a66949a", "discourse_text"
    ] = "This whole thing is point less how they have us in here for two days im missing my education. We could have finished this in one day and had the rest of the week to get back on the track of learning. I've missed both days of weight lifting, algebra, and my world history that i do not want to fail again! If their are any people actually gonna sit down and take the time to read this then\n\nDO NOT DO THIS NEXT YEAR\n\n.\n\nThey are giving us cold lunches. ham and cheese and an apple, I am 16 years old and my body needs proper food. I wouldnt be complaining if they served actual breakfast. but because of Michelle Obama and her healthy diet rule they surve us 1 poptart in the moring. How does the school board expect us to last from 7:05-12:15 on a pop tart? then expect us to get A's, we are more focused on lunch than anything else. I am about done so if you have the time to read this even though this does not count. Bring PROPER_NAME a big Mac from mc donalds, SCHOOL_NAME, (idk area code but its in LOCATION_NAME)       \xa0    "
    return df


def create_token_classification_df(
    df: pd.DataFrame, is_test: bool = False
) -> pd.DataFrame:
    fold_column_exist = "fold" in df.columns
    label_O = LABEL_O_WHEN_TEST if is_test else "O"

    all_obs = []
    for name, gr in tqdm(df.groupby("essay_id", sort=False)):
        essay_text_start_end = gr["essay_text"].values[0]
        token_labels = []
        token_obs = []

        end_pos = 0
        for idx, row in gr.reset_index(drop=True).iterrows():
            target_text = row["discourse_type"] + " " + row["discourse_text"].strip()
            essay_text_start_end = essay_text_start_end[
                :end_pos
            ] + essay_text_start_end[end_pos:].replace(
                row["discourse_text"].strip(), target_text, 1
            )

            start_pos = essay_text_start_end[end_pos:].find(target_text)
            if start_pos == -1:
                raise ValueError()
            start_pos += end_pos

            if idx == 0 and start_pos > 0:
                token_labels.append(label_O)
                token_obs.append(essay_text_start_end[:start_pos])

            if start_pos > end_pos and end_pos > 0:
                token_labels.append(label_O)
                token_obs.append(essay_text_start_end[end_pos:start_pos])

            end_pos = start_pos + len(target_text)
            token_labels.append(0 if is_test else row["discourse_effectiveness"])
            token_obs.append(essay_text_start_end[start_pos:end_pos])

            if idx == len(gr) - 1 and end_pos < len(essay_text_start_end):
                token_labels.append(label_O)
                token_obs.append(essay_text_start_end[end_pos:])

        all_obs.append(
            (name, token_labels, token_obs, row["fold"])
            if fold_column_exist
            else (name, token_labels, token_obs)
        )

    return pd.DataFrame(
        all_obs,
        columns=["essay_id", "tokens", "essay_text", "fold"]
        if fold_column_exist
        else ["essay_id", "tokens", "essay_text"],
    )


def load_train_df(path: PATH) -> pd.DataFrame:
    if Path(path).suffix == ".pq":
        train_df = pd.read_parquet(path)
    else:
        train_df = pd.read_csv(path)
    return train_df


def split_train_val(
    train_df: pd.DataFrame, fold: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "fold" in train_df.columns:
        if fold == -1:
            val_df = train_df[train_df["fold"] == 0].copy()
        else:
            val_df = train_df[train_df["fold"] == fold].copy()
            train_df = train_df[train_df["fold"] != fold].copy()
    else:
        val_df = train_df.copy()
    return train_df, val_df


def change_df_for_debug(train_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame([train_df.iloc[0, :].values for _ in range(len(train_df))])
    df.columns = train_df.columns
    return df


class CustomDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        config: Config,
        mode: str,
        label_encoder: Optional[LabelEncoder] = None,
        map_hugging_face_model_name_to_kaggle_dataset: bool = False,
    ) -> None:
        self.df = df.copy()
        self.config = config
        self.mode = mode
        self.label_encoder = label_encoder

        if mode == "train" and not config.training.is_pseudo:
            self.label_encoder = LabelEncoder().fit(
                np.concatenate(self.df[config.dataset.label_columns].values)
            )

        if self.mode != "test" and not config.training.is_pseudo:
            self.df[config.dataset.label_columns] = self.df[
                config.dataset.label_columns
            ].map(lambda labels: self.label_encoder.transform(labels))
            print(
                f"Example encoded label: {self.df[config.dataset.label_columns].values[0]}"
            )

        backbone = (
            HUGGING_FACE_MODEL_NAME_TO_KAGGLE_DATASET[config.architecture.backbone]
            if map_hugging_face_model_name_to_kaggle_dataset
            else config.architecture.backbone
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            backbone,
            add_prefix_space=True,
            use_fast=True,
            cache_dir=backbone
            if map_hugging_face_model_name_to_kaggle_dataset
            else None,
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
                    _ignore = (
                        LABEL_O_WHEN_TEST
                        if self.mode == "test"
                        else self.config.dataset.num_classes
                    )
                    if self.labels[idx][lab_idx] != _ignore:
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
        if self.mode != "test":
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


def preprocess_test_df(df: pd.DataFrame, dir: PATH) -> pd.DataFrame:
    df = add_essay_text_column(df, dir, is_test=True)
    df = modify_discourse_text(df)

    df["count"] = df["essay_text"].apply(len)
    df["orig_order"] = range(len(df))
    df = df.sort_values(
        ["count", "essay_id", "orig_order"], ascending=True
    ).reset_index(drop=True)

    return df
