COMPETITION_ABBREVIATION = "fpe"
LABEL_O_WHEN_TEST = 1


class _FILENAME:
    TRAIN_FOLDED = "train_folded.csv"
    TOKEN_CLASSIFICATION = "feedback_text_token_classification_v5.pq"

    FEEDBACK_PRIZE_2021_TRAIN_EXCEPT_EFFECTIVE = "old_competition_data.csv"
    FEEDBACK_PRIZE_2021_FORMATTED_TRAIN = "feedback_2021_pretrain.pq"

    CHECKPOINT = "checkpoint.pth"

    SUBMISSION = "submission.csv"

    def __init__(self) -> None:
        pass


FILENAME = _FILENAME()
