import json
from pathlib import Path
from typing import Union
import os


def increment_kernel_metadata(filepath: Union[str, Path]) -> None:
    with open(filepath, "r") as f:
        metadata = json.load(f)

    print(metadata)
    number = metadata["id"].split("-")[-1]
    print(number)
    print("-".join(metadata["id"].split("-")[:-1]) + "-")
    metadata["id"] = (
        "-".join(metadata["id"].split("-")[:-1]) + "-" + str(int(number) + 1)
    )
    metadata["title"] = (
        "-".join(metadata["title"].split("-")[:-1]) + "-" + str(int(number) + 1)
    )
    print(metadata)

    with open(filepath, "w") as f:
        json.dump(metadata, f)


if __name__ == "__main__":
    filepath = "./kaggle_kernels/train/kernel-metadata.json"

    increment_kernel_metadata(filepath)

    os.system("kaggle kernels push -p ./kaggle_kernels/train")