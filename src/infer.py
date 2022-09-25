from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from args import prepare_args, prepare_parser
from config import load_config, load_ensemble_configs
from const import FILENAME
from data import preprocess_test_df, create_token_classification_df, CustomDataset
from data_es import EssayDataset
from models.base import get_model
from trainer.trainer_native_pytorch import TrainerNativePytorch
from utils.env import set_environ
from utils.kaggle import create_submission
from utils.types import PATH


def infer(
    model_saved_dir: PATH,
    checkpoint_filename: str,
    df: pd.DataFrame,
    map_hugging_face_model_name_to_kaggle_dataset: bool,
) -> np.ndarray:
    config = load_config(model_saved_dir)

    if config.dataset.dataset_class == "feedback_dataset":
        df = create_token_classification_df(df, is_test=True)
        test_dataset = CustomDataset(
            df,
            config,
            "test",
            map_hugging_face_model_name_to_kaggle_dataset=map_hugging_face_model_name_to_kaggle_dataset,
        )
    elif config.dataset.dataset_class == "feedback_dataset_essay_ds":
        test_dataset = EssayDataset(
            df,
            config,
            "test",
            map_hugging_face_model_name_to_kaggle_dataset=map_hugging_face_model_name_to_kaggle_dataset,
        )

    trainer = TrainerNativePytorch(
        config,
        model_init=partial(
            get_model,
            model_class=config.architecture.model_class,
            is_test=True,
            map_hugging_face_model_name_to_kaggle_dataset=map_hugging_face_model_name_to_kaggle_dataset,
        ),
    )
    preds = trainer.predict(test_dataset, Path(model_saved_dir) / checkpoint_filename)
    return np.concatenate(preds, axis=0)


if __name__ == "__main__":
    set_environ()
    args = prepare_args(prepare_parser())
    ensemble_configs = load_ensemble_configs(args.ensemble_config_path)
    # ensemble_configs = load_ensemble_configs(
    #     "config/ensemble/axiomatic-vulture-ff.yaml"
    # )
    config = load_config(
        ensemble_configs[0].model_saved_dir
    )  # This config only used to access base config

    raw_df = pd.read_csv(
        Path(config.base.feedback_prize_effectiveness_dir) / "test.csv"
    )
    df = preprocess_test_df(raw_df, config.base.feedback_prize_effectiveness_dir)
    print(df)

    preds = []
    for _c in ensemble_configs:
        print(f"{_c.model_saved_dir} start")
        preds.append(
            infer(
                _c.model_saved_dir,
                _c.checkpoint_filename,
                df,
                args.map_hugging_face_model_name_to_kaggle_dataset,
            )
        )
    preds = np.mean(preds, axis=0)

    submission = create_submission(df["discourse_id"].values, preds)
    print(submission)
    submission.to_csv(FILENAME.SUBMISSION, index=False)
