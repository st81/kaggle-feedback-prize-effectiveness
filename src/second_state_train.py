import json
from pathlib import Path
from typing import Tuple
import random
import os
import datetime
import pytz

import numpy as np
import pandas as pd
import lightgbm
from sklearn import metrics
from tqdm import tqdm

from args import prepare_args, prepare_parser
from config import load_config
from const import FILENAME


def gen_x(values):
    range = 1
    return np.histogram(
        np.clip(values, 0.001, 0.999 * range), bins=3, density=True, range=(0, range)
    )[0]


class TabularModel:
    def read_data(self, config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_raw = pd.read_csv(config["train_path"])
        return train_raw

    def prepare_features(
        self,
        train_raw: pd.DataFrame,
        train_targets: np.array,
        train_folds: np.array,
        config: dict,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        df = train_raw.copy()
        return (df, config)

    def metric(self, y_true: np.array, y_pred: np.array) -> float:
        raise NotImplementedError

    def fit_predict(
        self, config, X_train, y_train, X_valid, y_valid
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def preprocess_target(self, config: dict, target: np.array) -> np.array:
        return target

    def postprocess_predictions(self, config: dict, preds: np.ndarray) -> np.ndarray:
        return preds

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)

    def run_cv(self, config: dict, fold=None, save_results: bool = True) -> float:
        timetag = (
            pytz.utc.localize(datetime.datetime.utcnow())
            .astimezone(pytz.timezone("Europe/Vienna"))
            .strftime("%m%d_%H%M%S")
        )

        self.set_seed(config["seed"])
        train_raw = self.read_data(config)

        if (fold is not None) and (fold not in train_raw[config["fold_column"]]):
            print(f"Fold {fold} not found in the data")
            return None

        train, config = self.prepare_features(train_raw, None, None, config)
        train_targets = train[config["target_column"]].copy()
        train_folds = train[config["fold_column"]].copy()
        to_out = train.copy()
        train = train.drop(
            [config["fold_column"], config["target_column"], "discourse_id"], axis=1
        )

        features = list(train.columns)
        train = train[pd.notnull(train_targets)]

        validation_oof = np.zeros((len(train), config["n_classes"]))
        results = {}

        if fold is None:
            output_folder = (
                f"{config['output_folder']}/{type(self).__name__}/cv/{timetag}"
            )
        else:
            output_folder = (
                f"{config['output_folder']}/{type(self).__name__}/fold_{fold}/{timetag}"
            )

        os.makedirs(output_folder, exist_ok=True)

        for _fold in np.sort(train_folds.unique()):
            if (fold is not None) and (_fold != fold):
                continue
            if config["print_progress"]:
                print("*" * 20 + f"FOLD: {_fold}" + "*" * 20)

            X_train = train[train_folds != _fold].copy()
            y_train = train_targets[train_folds != _fold]
            X_valid = train[train_folds == _fold].copy()
            y_valid = train_targets[train_folds == _fold]

            gbm, pred_valid = self.fit_predict(
                config,
                X_train[features],
                self.preprocess_target(config, y_train),
                X_valid[features],
                self.preprocess_target(config, y_valid),
            )

            gbm.save_model(
                f"{output_folder}/model_fold_{_fold}.txt",
                num_iteration=config["model_params"]["n_estimators"],
            )
            pred_valid = self.postprocess_predictions(config, pred_valid)

            if config["n_classes"] == 1:
                pred_valid = pred_valid.reshape(-1, 1)
            validation_oof[train_folds == _fold] = pred_valid

            fold_metric = self.metric(y_valid, pred_valid)
            results[str(_fold)] = fold_metric
            if config["print_progress"]:
                print(f"Metric for fold {_fold}: {fold_metric:5f}")

        if fold is None:
            oof_metric = self.metric(train_targets, validation_oof, train_folds)
            if config["print_progress"]:
                print(f"OOF metric: {oof_metric:5f}")
            results["oof"] = oof_metric
            output_metric = oof_metric
        else:
            output_metric = fold_metric

        results["features"] = features
        if save_results:
            np.save(
                f"{output_folder}/validation_oof_preds.npy", np.array(validation_oof)
            )
            with open(f"{output_folder}/config.json", "w") as fp:
                json.dump(config, fp, indent=4)
            with open(f"{output_folder}/results.json", "w") as fp:
                json.dump(results, fp, indent=4)
            print(f"Results saved to {output_folder}")

        return output_metric, output_folder, to_out


class LGBM(TabularModel):
    def fit_predict(
        self, config, X_train, y_train, X_valid, y_valid
    ) -> Tuple[np.ndarray, np.ndarray]:
        train_data = lightgbm.Dataset(X_train, label=y_train, params={"verbose": -1})
        if (X_valid is not None) and (y_valid is not None):
            valid_data = lightgbm.Dataset(
                X_valid, label=y_valid, params={"verbose": -1}
            )
        else:
            valid_data = None

        if (valid_data is None) or (config["model_params"].get("verbose_eval", 1) < 0):
            gbm = lightgbm.train(
                {
                    k: v
                    for k, v in config["model_params"].items()
                    if k not in ["verbose_eval", "n_estimators"]
                },
                train_data,
                num_boost_round=config["model_params"].get("n_estimators", 1000),
            )
        else:
            gbm = lightgbm.train(
                {
                    k: v
                    for k, v in config["model_params"].items()
                    if k not in ["verbose_eval", "n_estimators"]
                },
                train_data,
                num_boost_round=config["model_params"].get("n_estimators", 1000),
                valid_sets=valid_data,
                verbose_eval=config["model_params"].get("verbose_eval", "warn"),
            )

        if X_valid is None:
            valid_pred = None
        else:
            valid_pred = gbm.predict(X_valid)

        return gbm, valid_pred

    def metric(self, y_true: np.array, y_pred: np.array, train_folds=None) -> float:
        print(metrics.log_loss(y_true, y_pred))

        y = np.zeros_like(y_pred)

        for ii, jj in enumerate([x for x in y_true]):
            y[ii, jj] = 1

        pp2 = y_pred.copy()
        for _ in range(100):
            pp2 = pp2 * (y.mean(axis=0) / pp2.mean(axis=0))
            pp2 = pp2 / pp2.sum(axis=1, keepdims=True)

        if train_folds is not None:
            for f in range(5):
                print(
                    f"FOLD: {f}: {metrics.log_loss(y_true[train_folds == f], pp2[train_folds == f])}"
                )

        return metrics.log_loss(y_true, pp2)

    def prepare_features(
        self,
        train_raw: pd.DataFrame,
        train_targets: np.array,
        train_folds: np.array,
        config: dict,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        from sklearn.preprocessing import LabelEncoder

        df = train_raw.copy()

        df = df.drop(
            ["discourse_text", "essay_text", "essay_id", "discourse_effectiveness"],
            axis=1,
        )
        print(df.columns)
        return (df, config)


if __name__ == "__main__":
    args = prepare_args(prepare_parser())
    args.config_path = "config/axiomatic-vulture-ff.yaml"
    config = load_config(args.config_path)

    input_data_dir = Path(config.base.input_data_dir)
    # oof = pd.read_csv(input_data_dir / FILENAME.OOF_AFTER_SCALING)
    # tt = pd.read_csv(input_data_dir / FILENAME.TRAIN_FOLDED)

    # oof = tt[["discourse_id", "fold", "discourse_text", "essay_text"]].merge(oof)
    # print(oof)
    # mapping = {"Adequate": 0, "Effective": 1, "Ineffective": 2}
    # oof["label"] = oof["discourse_effectiveness"].map(mapping)

    # df = oof.copy()

    # all_groups = []

    # gb = df.groupby("essay_id", sort=False)
    # for name, group in tqdm(gb):
    #     group["n_types"] = group["discourse_type"].nunique()
    #     for class_name in ["Adequate", "Effective", "Ineffective"]:

    #         if class_name in ["Adequate", "Effective"]:
    #             continue
    #         for idx, val in enumerate(gen_x(group[class_name].values)):
    #             group[f"{class_name}_bin_{idx}"] = val
    #         group[f"mean_{class_name}"] = group[class_name].mean()

    #     all_groups.append(group)

    # df = pd.concat(all_groups).reset_index(drop=True)

    # disc_types_mapping = {
    #     "Lead": 0,
    #     "Position": 1,
    #     "Claim": 2,
    #     "Evidence": 3,
    #     "Counterclaim": 4,
    #     "Rebuttal": 5,
    #     "Concluding Statement": 6,
    # }
    # df["len_disc"] = df["discourse_text"].str.len()

    # df["discourse_type"] = df["discourse_type"].map(disc_types_mapping)

    # df["paragraph_cnt"] = df["essay_text"].map(lambda x: len(x.split("\n\n")))

    # df.to_csv(input_data_dir / FILENAME.TRAIN_LGB, index=False)

    config_train = dict(
        train_path=str(input_data_dir / FILENAME.TRAIN_LGB),
        fold_column="fold",
        target_column="label",
        n_classes=3,
        regression=False,
        seed=1337,
        output_folder="output/lightgbm/results",
        print_progress=True,
        data_params={},
        model_params={
            "metric": "multi_logloss",
            "num_classes": 3,
            "boosting_type": "gbdt",
            "objective": "multiclass",
            "n_estimators": 200,
            "learning_rate": 0.1,
            "num_leaves": 4,
            "seed": 1337,
            "verbose": -1,
            "num_threads": 40,
            "early_stopping_round": -1,
            "verbose_eval": 50,
        },
    )
    print(config_train)

    model = LGBM()
    model.run_cv(config_train, fold=None)
