from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

from args import prepare_args, prepare_parser
from config import load_config
from const import FILENAME
from data import modify_discourse_text, create_token_classification_df
from utils.types import PATH


class FeedbackPrizeEffectivenessData:
    def __init__(self, data_dir: PATH, save_dir: PATH) -> None:
        self.data_dir = data_dir
        self.save_dir = save_dir
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        self.train_df = pd.read_csv(Path(self.data_dir) / "train.csv")

    @property
    def train_folded_path(self) -> str:
        return Path(self.save_dir) / FILENAME.TRAIN_FOLDED

    @property
    def token_classification_path(self) -> str:
        return Path(self.save_dir) / FILENAME.TOKEN_CLASSIFICATION

    def prepare_folded(self) -> None:
        for fold, (_, val_idx) in enumerate(
            list(
                StratifiedGroupKFold(n_splits=5).split(
                    np.arange(len(self.train_df)),
                    self.train_df["discourse_effectiveness"],
                    groups=self.train_df["essay_id"],
                )
            )
        ):
            self.train_df.loc[val_idx, "fold"] = fold
        self.train_df["fold"] = self.train_df["fold"].astype(int)
        essay_texts = {}
        for filename in Path(self.data_dir).glob("train/*.txt"):
            with open(filename, "r") as f:
                lines = f.read()
            essay_texts[filename.stem] = lines
        self.train_df["essay_text"] = self.train_df["essay_id"].map(essay_texts)

        discourse_effectiveness_ohe = pd.get_dummies(
            self.train_df["discourse_effectiveness"]
        )
        discourse_effectiveness_ohe.columns = [
            f"discourse_effectiveness_{c}" for c in discourse_effectiveness_ohe.columns
        ]
        self.train_df = pd.concat([self.train_df, discourse_effectiveness_ohe], axis=1)

        self.train_df.to_csv(Path(self.save_dir) / FILENAME.TRAIN_FOLDED, index=False)

    def prepare_token_classification(self) -> None:
        df = pd.read_csv(self.train_folded_path)
        df = modify_discourse_text(df)

        all_obs = []
        for name, gr in tqdm(df.groupby("essay_id", sort=False)):
            essay_text_start_end = gr["essay_text"].values[0]
            token_labels = []
            token_obs = []

            end_pos = 0
            for idx, row in gr.reset_index(drop=True).iterrows():
                target_text = (
                    row["discourse_type"] + " " + row["discourse_text"].strip()
                )
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
                    token_labels.append("O")
                    token_obs.append(essay_text_start_end[:start_pos])

                if start_pos > end_pos and end_pos > 0:
                    token_labels.append("O")
                    token_obs.append(essay_text_start_end[end_pos:start_pos])

                end_pos = start_pos + len(target_text)
                token_labels.append(row["discourse_effectiveness"])
                token_obs.append(essay_text_start_end[start_pos:end_pos])

                if idx == len(gr) - 1 and end_pos < len(essay_text_start_end):
                    token_labels.append("O")
                    token_obs.append(essay_text_start_end[end_pos:])
            all_obs.append((name, token_labels, token_obs, row["fold"]))

        pd.DataFrame(
            all_obs, columns=["essay_id", "tokens", "essay_text", "fold"]
        ).to_parquet(self.token_classification_path, index=False)


class FeedbackPrize2021Data:
    def __init__(self, data_dir: PATH, save_dir: PATH) -> None:
        self.data_dir = data_dir
        self.save_dir = save_dir
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        self.train_df = pd.read_csv(Path(self.data_dir) / "train.csv")

    def prepare_old_comp_data(self, train_folded_path: PATH) -> None:
        train_df = self.train_df.rename(columns={"id": "essay_id"})
        train_folded = pd.read_csv(train_folded_path)
        train_df = train_df[
            ~train_df["essay_id"].isin(train_folded["essay_id"])
        ].reset_index(drop=True)

        essay_texts = {}
        for filename in Path(self.data_dir).glob("train/*.txt"):
            with open(filename, "r") as f:
                lines = f.read()
            essay_texts[filename.stem] = lines
        train_df["essay_text"] = train_df["essay_id"].map(essay_texts)
        train_df.to_csv(
            Path(self.save_dir) / FILENAME.FEEDBACK_PRIZE_2021_TRAIN_EXCEPT_EFFECTIVE,
            index=False,
        )

    def prepare(self, old_comp_data_path: PATH) -> None:
        df = pd.read_csv(old_comp_data_path)

        all_obs = []
        for name, _df in tqdm(df.groupby("essay_id", sort=False)):
            essay_text_start_end: str = _df["essay_text"].values[0]
            token_labels = []
            token_obs = []

            end_pos = 0
            for idx, row in _df.reset_index(drop=True).iterrows():
                target_text = row["discourse_text"].strip()
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
                    token_labels.append("O")
                    token_obs.append(essay_text_start_end[:start_pos])

                if start_pos > end_pos and end_pos > 0:
                    token_labels.append("O")
                    token_obs.append(essay_text_start_end[end_pos:start_pos])

                end_pos = start_pos + len(target_text)
                token_labels.append("A" + row["discourse_type"])
                token_obs.append(essay_text_start_end[start_pos:end_pos])

                if idx == len(_df) - 1 and end_pos < len(essay_text_start_end):
                    token_labels.append("O")
                    token_obs.append(essay_text_start_end[end_pos:])

            all_obs.append((name, token_labels, token_obs))

        pd.DataFrame(all_obs, columns=["essay_id", "tokens", "essay_text"]).to_parquet(
            Path(self.save_dir) / FILENAME.FEEDBACK_PRIZE_2021_FORMATTED_TRAIN,
            index=False,
        )


class PseudoData:
    def __init__(
        self, pseudo_label_path: PATH, old_comp_data_path: PATH, save_dir: PATH
    ) -> None:
        labels = pd.read_csv(pseudo_label_path)
        self.df = pd.read_csv(old_comp_data_path).merge(labels)
        print(self.df)
        self.save_dir = save_dir
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        self.save_filename = "_".join(pseudo_label_path.split("_")[:3]) + ".pq"

    def prepare(self) -> None:
        token_classification_df = create_token_classification_df(
            self.df, is_pseudo=True
        )
        print(token_classification_df)
        token_classification_df.to_parquet(Path(self.save_dir) / self.save_filename)


if __name__ == "__main__":
    args = prepare_args(prepare_parser())
    # config = load_config(args.config_path)
    config = load_config("config/pretrain_2021.yaml")

    # feedback_prize_effectiveness_data = FeedbackPrizeEffectivenessData(
    #     config.base.feedback_prize_effectiveness_dir, config.base.input_data_dir
    # )
    # feedback_prize_effectiveness_data.prepare_folded()
    # feedback_prize_effectiveness_data.prepare_token_classification()

    # feedback_prize_2021_data = FeedbackPrize2021Data(
    #     config.base.feedback_prize_2021_dir, config.base.input_data_dir
    # )
    # feedback_prize_2021_data.prepare_old_comp_data(
    #     Path(config.base.input_data_dir) / FILENAME.TRAIN_FOLDED
    # )
    # feedback_prize_2021_data.prepare(
    #     Path(config.base.input_data_dir)
    #     / FILENAME.FEEDBACK_PRIZE_2021_TRAIN_EXCEPT_EFFECTIVE
    # )

    pseudo_data = PseudoData(
        "pseudo_75_ff_raw.csv",
        Path(config.base.input_data_dir)
        / FILENAME.FEEDBACK_PRIZE_2021_TRAIN_EXCEPT_EFFECTIVE,
        config.base.input_data_dir,
    ).prepare()
