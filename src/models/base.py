from config import Config
from models.feedback_model import Net


def get_model(
    config: Config,
    is_test: bool = False,
    map_hugging_face_model_name_to_kaggle_dataset: bool = False,
) -> Net:
    return Net(config, is_test, map_hugging_face_model_name_to_kaggle_dataset)
