from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from args import prepare_args, prepare_parser
from config import load_config, load_ensemble_configs
from const import FILENAME
from data import preprocess_test_df, create_token_classification_df, CustomDataset
from models.base import get_model
from trainer.trainer_native_pytorch import TrainerNativePytorch
from utils.kaggle import create_submission
from utils.types import PATH


def infer(
    model_saved_dir: PATH,
    df: pd.DataFrame,
    map_hugging_face_model_name_to_kaggle_dataset: bool,
) -> np.ndarray:
    config = load_config(model_saved_dir)

    df = create_token_classification_df(df, is_test=True)
    test_dataset = CustomDataset(
        df,
        config,
        "test",
        map_hugging_face_model_name_to_kaggle_dataset=map_hugging_face_model_name_to_kaggle_dataset,
    )

    trainer = TrainerNativePytorch(
        config,
        model_init=partial(
            get_model,
            is_test=True,
            map_hugging_face_model_name_to_kaggle_dataset=map_hugging_face_model_name_to_kaggle_dataset,
        ),
    )
    preds = trainer.predict(test_dataset, model_saved_dir)
    return np.concatenate(preds, axis=0)


if __name__ == "__main__":
    args = prepare_args(prepare_parser())
    ensemble_configs = load_ensemble_configs(args.ensemble_config_path)
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
                df,
                args.map_hugging_face_model_name_to_kaggle_dataset,
            )
        )
    preds = np.mean(preds, axis=0)

    submission = create_submission(df["discourse_id"].values, preds)
    print(submission)
    submission.to_csv(FILENAME.SUBMISSION, index=False)
