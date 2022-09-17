from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

from args import prepare_args, prepare_parser
from config import load_config
from const import FILENAME
from utils.types import PATH


class FeedbackPrizeEffectivenessData:
    def __init__(self, data_dir: PATH, save_dir: PATH) -> None:
        self.data_dir = data_dir
        self.save_dir = save_dir
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        self.train_df = pd.read_csv(Path(self.data_dir) / "train.csv")

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
        self.train_df.to_csv(Path(self.save_dir) / FILENAME.TRAIN_FOLDED, index=False)


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


if __name__ == "__main__":
    args = prepare_args(prepare_parser())
    config = load_config(args.config_dir)
    config = load_config("config")

    FeedbackPrizeEffectivenessData(
        config.base.feedback_prize_effectiveness_dir, config.base.input_data_dir
    ).prepare_folded()

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
