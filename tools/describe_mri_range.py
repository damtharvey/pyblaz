import argparse
import pathlib

import torch
import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../data")
    args = parser.parse_args()

    data_path = pathlib.Path(args.data)
    example_paths = tuple((data_path / "lgg-mri-segmentation" / "as_tensors").glob("*"))

    minimum = float("inf")
    maximum = float("-inf")

    for example_path in tqdm.tqdm(example_paths):
        example = torch.load(example_path)
        minimum = min(minimum, example.min())
        maximum = max(maximum, example.max())

    print(f"{minimum=}, {maximum=}")


if __name__ == "__main__":
    main()
