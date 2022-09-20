import json
from pathlib import Path
import subprocess
from typing import List

import numpy as np
import pandas as pd

from utils.types import PATH


HUGGING_FACE_MODEL_NAME_TO_KAGGLE_DATASET = {
    "microsoft/deberta-large": "../input/debertalarge"
}


def create_kaggle_dataset_metadata(dataset_id: str, save_dir: PATH) -> None:
    d = {
        "title": dataset_id,
        "id": f"shutotakahashi/{dataset_id}",
        "licenses": [{"name": "CC0-1.0"}],
    }

    with open(f"{save_dir}/dataset-metadata.json", "w") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)


def create_kaggle_dataset(dataset_id: str, dataset_dir: str) -> None:
    create_kaggle_dataset_metadata(dataset_id, dataset_dir)
    subprocess.run(f"kaggle datasets create -p {dataset_dir} -r zip", shell=True)


def create_notebook_file(dataset_id: str, save_path: PATH) -> None:
    d = {
        "metadata": {
            "kernelspec": {
                "language": "python",
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
                "version": "3.6.4",
                "file_extension": ".py",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "name": "python",
                "mimetype": "text/x-python",
            },
        },
        "nbformat_minor": 4,
        "nbformat": 4,
        "cells": [
            {
                "cell_type": "code",
                "source": f"!cp -r ../input/{dataset_id}/* .",
                "metadata": {
                    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
                    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
                    "trusted": True,
                },
                "execution_count": None,
                "outputs": [],
            }
        ],
    }

    with open(save_path, "w") as f:
        json.dump(d, f, indent=4)


def create_kaggle_kernel_metadata(
    id: str,
    code_filename: str,
    enable_gpu: bool = False,
    dataset_sources: List[str] = [],
    competition_sources: List[str] = [],
    kernel_sources: List[str] = [],
    save_dir: PATH = ".",
) -> None:
    d = {
        "id": f"shutotakahashi/{id}",
        "title": id,
        "code_file": code_filename,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": enable_gpu,
        "enable_internet": "true",
        "dataset_sources": dataset_sources,
        "competition_sources": competition_sources,
        "kernel_sources": kernel_sources,
    }

    with open(Path(save_dir) / "kernel-metadata.json", "w") as f:
        json.dump(d, f, indent=4)


def create_kaggle_kernel(
    id: str,
    code_filename: str,
    enable_gpu: bool = False,
    dataset_sources: List[str] = [],
    competition_sources: List[str] = [],
    kernel_sources: List[str] = [],
    save_dir: PATH = ".",
) -> None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    create_notebook_file(id, Path(save_dir) / code_filename)
    create_kaggle_kernel_metadata(
        f"{id}-cp",
        code_filename,
        enable_gpu,
        dataset_sources,
        competition_sources,
        kernel_sources,
        save_dir,
    )
    subprocess.run(f"kaggle kernels push -p {save_dir}", shell=True)


def create_submission(discourse_ids: List[str], preds: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "discourse_id": discourse_ids,
            "Ineffective": preds[:, 2],
            "Adequate": preds[:, 0],
            "Effective": preds[:, 1],
        }
    )
