COMPETITION_ABBREVIATION = "fpe"
LABEL_O_WHEN_TEST = 1
LABEL_O_WHEN_PSEUDO = [-1, -1, -1]


class _FILENAME:
    TRAIN_FOLDED = "train_folded.csv"
    TOKEN_CLASSIFICATION = "feedback_text_token_classification_v5.pq"

    FEEDBACK_PRIZE_2021_TRAIN_EXCEPT_EFFECTIVE = "old_competition_data.csv"
    FEEDBACK_PRIZE_2021_FORMATTED_TRAIN = "feedback_2021_pretrain.pq"

    CHECKPOINT = "checkpoint.pth"

    SUBMISSION = "submission.csv"

    OOF_AFTER_SCALING = "oof_151_after_scaling.csv"
    OOF_AFTER_SCALING_IND_MODELS = "oof_151_after_scaling_ind_models.csv"
    FIRST_LVL_ENSEMBLE_NPY = "first_lvl_ensemble.npy"
    FIRST_LVL_ENSEMBLE_PKL = "first_lvl_ensemble.pkl"

    TRAIN_LGB = "train_lgb.csv"

    def __init__(self) -> None:
        pass

    def oof_filename(self, fold: int, seed: int) -> str:
        return f"oof_fold{fold}_seed{seed}.csv"


FILENAME = _FILENAME()
