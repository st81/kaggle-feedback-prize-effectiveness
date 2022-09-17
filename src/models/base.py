from config import Config
from models.feedback_model import Net


def get_model(config: Config) -> Net:
    return Net(config)
