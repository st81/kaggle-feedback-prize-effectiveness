from typing import Union
from config import Config
from models.feedback_model import Net
from models.feedback_essay_model import Net as EssayNet


def get_model(
    config: Config,
    is_test: bool = False,
    map_hugging_face_model_name_to_kaggle_dataset: bool = False,
    model_class: str = "feedback_model",
) -> Union[Net, EssayNet]:
    if model_class == "feedback_model":
        return Net(config, is_test, map_hugging_face_model_name_to_kaggle_dataset)
    elif model_class == "feedback_essay_model":
        return EssayNet(config, is_test, map_hugging_face_model_name_to_kaggle_dataset)
    else:
        return ValueError(
            "'config.architecture.model_class' must be either 'feedback_model' or 'feedback_essay_model'"
        )
