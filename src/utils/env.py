import os


def set_environ() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
