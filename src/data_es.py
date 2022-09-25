import collections
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from config import Config
from utils.kaggle import HUGGING_FACE_MODEL_NAME_TO_KAGGLE_DATASET


class EssayDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        config: Config,
        mode: str,
        map_hugging_face_model_name_to_kaggle_dataset: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.config = config
        self.mode = mode

        backbone = (
            HUGGING_FACE_MODEL_NAME_TO_KAGGLE_DATASET[config.architecture.backbone]
            if map_hugging_face_model_name_to_kaggle_dataset
            else config.architecture.backbone
        )
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.tokenizer.sep_token is None:
            self.tokenizer.sep_token = " "

        if len(self.config.dataset.separator):
            self.config._tokenizer_sep_token = self.config.dataset.separator
        else:
            self.config._tokenizer_sep_token = self.tokenizer.sep_token

        self.text = self.get_texts(self.df, self.config, self.tokenizer.sep_token)

        if self.mode != "test":
            self.label_cols = self.config.dataset.label_columns

            if self.label_cols == ["discourse_type"]:
                raise NotImplementedError
            else:
                self.labels = self.df[self.label_cols].values

            self.weights = self.labels.max(axis=1)

        if self.config.dataset.group_discourse:
            if self.mode != "test":
                self.df["weights"] = self.weights
                self.df.loc[
                    self.df.discourse_id == "56744a66949a", "discourse_text"
                ] = "This whole thing is point less how they have us in here for two days im missing my education. We could have finished this in one day and had the rest of the week to get back on the track of learning. I've missed both days of weight lifting, algebra, and my world history that i do not want to fail again! If their are any people actually gonna sit down and take the time to read this then\n\nDO NOT DO THIS NEXT YEAR\n\n.\n\nThey are giving us cold lunches. ham and cheese and an apple, I am 16 years old and my body needs proper food. I wouldnt be complaining if they served actual breakfast. but because of Michelle Obama and her healthy diet rule they surve us 1 poptart in the moring. How does the school board expect us to last from 7:05-12:15 on a pop tart? then expect us to get A's, we are more focused on lunch than anything else. I am about done so if you have the time to read this even though this does not count. Bring PROPER_NAME a big Mac from mc donalds, SCHOOL_NAME, (idk area code but its in LOCATION_NAME)       \xa0    "

            grps = self.df.groupby("essay_id", sort=False)
            self.grp_texts = []
            self.grp_labels = []
            self.grp_weights = []

            s = 0

            if self.mode == "train" and self.config.architecture.aux_type:
                raise NotImplementedError

            for jjj, grp in enumerate(grps.groups):
                g = grps.get_group(grp)
                t = g["essay_text"].values[0]

                labels = []
                labels_aux = []
                weights = []

                end = 0
                for j in range(len(g)):
                    if self.mode != "test":
                        labels.append(g[self.label_cols].values[j])
                        weights.append(g["weights"].values[j])

                    if self.mode == "train" and self.config.architecture.aux_type:
                        raise NotImplementedError

                    d = g["discourse_text"].values[j]

                    start = t[end:].find(d.strip())
                    if start == -1:
                        print("ERROR")

                    start = start + end

                    end = start + len(d.strip())

                    if self.config.architecture.use_sep:
                        raise NotImplementedError
                    elif self.config.architecture.aux_type:
                        raise NotImplementedError
                    elif self.config.architecture.use_type:
                        raise NotImplementedError
                    elif self.config.architecture.no_type:
                        raise NotImplementedError
                    else:
                        t = (
                            t[:start]
                            + f" [START] {g['discourse_type'].values[j]} "
                            + t[start:end]
                            + " [END] "
                            + t[end:]
                        )

                if self.config.dataset.add_group_types:
                    t = (
                        " ".join(g["discourse_type"].values)
                        + f" {self.config._tokenizer_sep_token} "
                        + t
                    )

                self.grp_texts.append(t)
                if self.mode != "test":
                    self.grp_labels.append(labels)
                    self.grp_weights.append(weights)
                if self.mode == "train" and self.config.architecture.aux_type:
                    raise NotImplementedError
                s += len(g)

                if self.mode != "test" and jjj == 0:
                    print(t)
                    print(labels)

            if self.config.dataset.group_discourse:
                self.config._tokenizer_start_token_id = []
                self.config._tokenizer_end_token_id = []

                if self.config.architecture.use_type:
                    raise NotImplementedError
                else:
                    self.tokenizer.add_tokens(["[START]", "[END]"], special_tokens=True)
                    idx = 1
                    self.config._tokenizer_start_token_id.append(
                        self.tokenizer.encode("[START]")[idx]
                    )
                    self.config._tokenizer_end_token_id.append(
                        self.tokenizer.encode("[END]")[idx]
                    )

                print(self.config._tokenizer_start_token_id)
                print(self.config._tokenizer_end_token_id)

            if self.config.tokenizer.add_newline_token:
                self.tokenizer.add_tokens(["\n"], special_tokens=True)

            self.config._tokenizer_size = len(self.tokenizer)
        else:
            print(self.text[0])

        self.config._tokenizer_cls_token_id = self.tokenizer.cls_token_id
        self.config._tokenizer_sep_token_id = self.tokenizer.sep_token_id
        self.config._tokenizer_mask_token_id = self.tokenizer.mask_token_id

    def __len__(self) -> int:
        if self.config.dataset.group_discourse:
            return len(self.grp_texts)
        else:
            raise NotImplementedError

    @staticmethod
    def collate_fn(batch: torch.Tensor) -> torch.Tensor:
        elem = batch[0]

        ret = {}
        for key in elem:
            if key in {"target", "weight"}:
                ret[key] = [d[key].float() for d in batch]
            elif key in {"target_aux"}:
                ret[key] = [d[key].float() for d in batch]
            else:
                ret[key] = torch.stack([d[key] for d in batch], 0)
        return ret

    @staticmethod
    def get_train_collate_fn(config: Config):
        if config.dataset.group_discourse:
            return EssayDataset.collate_fn
        else:
            raise NotImplementedError

    @staticmethod
    def get_val_collate_fn(config: Config):
        if config.dataset.group_discourse:
            return EssayDataset.collate_fn
        else:
            raise NotImplementedError

    @staticmethod
    def batch_to_device(batch, device):
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        elif isinstance(batch, collections.abc.Mapping):
            return {
                key: EssayDataset.batch_to_device(value, device)
                for key, value in batch.items()
            }
        elif isinstance(batch, collections.abc.Sequence):
            return [EssayDataset.batch_to_device(value, device) for value in batch]
        else:
            raise ValueError(f"Can not move {type(batch)} to device.")

    def get_texts(self, df: pd.DataFrame, config: Config, separator: str):
        if separator is None:
            raise NotImplementedError

        if isinstance(config.dataset.text_column, str):
            texts = df[config.dataset.text_column].astype(str)
            texts = texts.values
        else:
            columns = list(config.dataset.text_column)
            join_str = f" {separator} "
            texts = df[columns].astype(str)
            texts = texts.apply(lambda x: join_str.join(x), axis=1).values

        return texts

    def _read_data(self, idx: int, sample: Dict[str, Any]) -> Dict[str, Any]:
        if self.config.dataset.group_discourse:
            text = self.grp_texts[idx]
        else:
            raise NotImplementedError

        if idx == 0:
            print(text)

        sample.update(self.encode(text))
        return sample

    def encode(self, text):
        sample = dict()
        encodings = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.config.tokenizer.max_length,
        )
        sample["input_ids"] = encodings["input_ids"][0]
        sample["attention_mask"] = encodings["attention_mask"][0]
        return sample

    def _read_label(self, idx: int, sample: Dict[str, Any]) -> Dict[str, Any]:
        if self.config.dataset.group_discourse:
            sample["target"] = self.grp_labels[idx]
            if self.mode == "train" and self.config.architecture.aux_type:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return sample

    def __getitem__(self, idx: int):
        sample = dict()

        sample = self._read_data(idx, sample)

        if self.mode != "test":
            if self.label_cols is not None:
                sample = self._read_label(idx, sample)

            if "target" in sample:
                sample["target"] = torch.tensor(np.array(sample["target"])).float()

        return sample
