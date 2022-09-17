import datetime


def now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
