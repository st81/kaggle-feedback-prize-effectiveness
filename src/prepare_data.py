from argparse import ArgumentParser
from pathlib import Path
import pickle
from typing import List

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import log_loss
from tqdm import tqdm

from args import prepare_args, prepare_parser
from config import OofConfig, load_config, load_prepare_data_config
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
        self,
        pseudo_label_path: PATH,
        train_folded_path: PATH,
        old_comp_data_path: PATH,
        save_dir: PATH,
    ) -> None:
        labels = pd.read_csv(pseudo_label_path)
        self.df = pd.read_csv(train_folded_path)
        self.df_old = pd.read_csv(old_comp_data_path).merge(labels)
        self.save_dir = save_dir
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        self.save_filename = "_".join(pseudo_label_path.split("_")[:3]) + ".pq"

    def prepare(self) -> None:
        token_classification_df = create_token_classification_df(
            self.df_old, is_pseudo=True
        )
        print(token_classification_df)
        token_classification_df.to_parquet(Path(self.save_dir) / self.save_filename)

    def prepare_pseudo_only(self):
        pass


class OofEnsemble:
    def __init__(
        self, oof_configs: List[OofConfig], train_folded_path: PATH, save_dir: PATH
    ) -> None:
        self.oof_configs = oof_configs
        self.orig = pd.read_csv(train_folded_path).set_index("discourse_id")
        print(self.orig)
        self.save_dir = Path(save_dir)
        self.label_cols = ["Adequate", "Effective", "Ineffective"]

    def prepare(self) -> None:
        preds = []
        for oof_config in self.oof_configs:
            pps = []

            for oof_saved_dir in oof_config.oof_saved_dirs:
                vs = []
                for p in Path(oof_saved_dir).glob("*.csv"):
                    v = pd.read_csv(p)
                    vs.append(v)

                v = pd.concat(vs)
                v = v.set_index("discourse_id")
                v = v.loc[self.orig.index]
                pps.append(v[self.label_cols].values)

            pps = np.mean(pps, axis=0)

            preds.append(pps)

        y = np.zeros_like(preds[0])
        for ii, jj in enumerate(
            [
                self.label_cols.index(x)
                for x in self.orig["discourse_effectiveness"].values
            ]
        ):
            y[ii, jj] = 1
        ps = np.array(preds).copy()

        def scale_probs(pp_single):
            pp = pp_single.copy()

            for _ in range(100):
                pp = pp * (y.mean(axis=0).reshape(1, 3) / pp.mean(axis=0))
                pp = pp / pp.sum(axis=1, keepdims=True)

            return pp

        for i, ppp in enumerate(ps):
            preds[i] = scale_probs(ppp)

        def weights_tune(weights, preds):
            pp = np.average(preds, axis=0, weights=weights)

            eps = 0.0001
            pp = pp.clip(eps, 1 - eps)
            pp = pp / pp.sum(axis=1, keepdims=True)

            pp2 = pp.copy()
            for _ in range(10):
                pp2 = pp2 * (y.mean(axis=0) / pp2.mean(axis=0))
                pp2 = pp2 / pp2.sum(axis=1, keepdims=True)
            pp = pp2

            err = log_loss(y, pp)
            return err

        weights_init = [1] * len(preds)
        res = minimize(
            weights_tune, weights_init, args=(preds), method="Nelder-Mead", tol=1e-6
        )
        print("Optimized weights: ", res.x)
        weights = res.x

        pp = np.average(preds, axis=0, weights=weights)

        eps = 0.0001
        pp = pp.clip(eps, 1 - eps)
        pp = pp / pp.sum(axis=1, keepdims=True)

        pp = scale_probs(pp)

        y = np.zeros_like(pp)
        for ii, jj in enumerate(
            [
                self.label_cols.index(x)
                for x in self.orig["discourse_effectiveness"].values
            ]
        ):
            y[ii, jj] = 1

        print(log_loss(y, pp))

        df = self.orig[["essay_id", "discourse_type", "discourse_effectiveness"]].copy()
        df["Adequate"] = pp[:, 0]
        df["Effective"] = pp[:, 1]
        df["Ineffective"] = pp[:, 2]
        df.to_csv(self.save_dir / FILENAME.OOF_AFTER_SCALING)

        for model_id, model_pred in enumerate(preds):
            df[f"Adequate_{model_id}"] = model_pred[:, 0]
            df[f"Effective_{model_id}"] = model_pred[:, 1]
            df[f"Ineffective_{model_id}"] = model_pred[:, 2]
        df.to_csv(self.save_dir / FILENAME.OOF_AFTER_SCALING_IND_MODELS)

        np.save(self.save_dir / FILENAME.FIRST_LVL_ENSEMBLE_NPY, pp)
        with open(self.save_dir / FILENAME.FIRST_LVL_ENSEMBLE_PKL, "wb") as f:
            pickle.dump(preds, f)


def _add_args(parent_parser: ArgumentParser) -> ArgumentParser:
    parser = parent_parser.add_argument_group("prepare_data_args")
    parser.add_argument("--prepare_data_types", nargs="+", default=[])
    parser.add_argument("--prepare_data_config_path", type=str)
    return parent_parser


if __name__ == "__main__":
    args = prepare_args(_add_args(prepare_parser()))
    prepare_data_config = load_prepare_data_config(args.prepare_data_config_path)
    config = load_config(args.config_path)
    # config = load_config("config/pretrain_2021.yaml")

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

    if "pseudo_data" in prepare_data_config.prepare_data_types:
        PseudoData(
            "pseudo_75_ff_raw.csv",
            Path(config.base.input_data_dir) / FILENAME.TRAIN_FOLDED,
            Path(config.base.input_data_dir)
            / FILENAME.FEEDBACK_PRIZE_2021_TRAIN_EXCEPT_EFFECTIVE,
            config.base.input_data_dir,
        ).prepare()

    if "oof_ensemble" in prepare_data_config.prepare_data_types:
        OofEnsemble(
            prepare_data_config.oof_configs,
            Path(config.base.input_data_dir) / FILENAME.TRAIN_FOLDED,
            config.base.input_data_dir,
        ).prepare()
