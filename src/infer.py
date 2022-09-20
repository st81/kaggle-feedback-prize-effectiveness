from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from args import prepare_args, prepare_parser
from config import Config, load_config
from const import FILENAME
from data import preprocess_test_df, create_token_classification_df, CustomDataset
from models.base import get_model
from trainer.trainer_native_pytorch import TrainerNativePytorch
from utils.kaggle import create_submission
from utils.types import PATH


def infer(
    config: Config,
    df: pd.DataFrame,
    model_saved_dir: PATH,
    map_hugging_face_model_name_to_kaggle_dataset: bool,
) -> np.ndarray:
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
    config = load_config(list(Path(args.model_saved_dir).glob("*.yaml"))[0])

    raw_df = pd.read_csv(
        Path(config.base.feedback_prize_effectiveness_dir) / "test.csv"
    )
    df = preprocess_test_df(raw_df, config.base.feedback_prize_effectiveness_dir)
    print(df)
    preds = infer(
        config,
        df,
        args.model_saved_dir,
        args.map_hugging_face_model_name_to_kaggle_dataset,
    )

    submission = create_submission(df["discourse_id"].values, preds)
    print(submission)
    submission.to_csv(Path(config.base.output_dir) / FILENAME.SUBMISSION, index=False)
