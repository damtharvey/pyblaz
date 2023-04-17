import argparse
import pathlib

import torch
import tqdm

from pyblaz import compression


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-directory", default="data/ShallowWatersEquations")
    parser.add_argument("--results-directory", default="results/ShallowWaters")
    args = parser.parse_args()

    data_directory = pathlib.Path(args.data_directory)
    results_directory = pathlib.Path(args.results_directory)
    results_directory.mkdir(parents=True, exist_ok=True)

    dtype = torch.float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    compressor = compression.PyBlaz(block_shape=(128, 128), dtype=dtype, device=device)

    to_write = ["fastmath_vs_o3,ftz_vs_o3"]

    for time_step in tqdm.tqdm(range(500)):
        o3 = torch.tensor(read_frame(data_directory, "output", time_step), dtype=dtype, device=device)
        fastmath = torch.tensor(read_frame(data_directory, "fastmath_output", time_step), dtype=dtype, device=device)
        ftz = torch.tensor(read_frame(data_directory, "ftz", time_step), dtype=dtype, device=device)

        o3_sum = weighted_sum(compressor.blockwise_transform(pad(o3)[None, None, ...]))
        fastmath_sum = weighted_sum(compressor.blockwise_transform(pad(fastmath)[None, None, ...]))
        ftz_sum = weighted_sum(compressor.blockwise_transform(pad(ftz)[None, None, ...]))

        to_write.append(f"{(fastmath_sum - o3_sum).abs().item()},{(ftz_sum - o3_sum).abs().item()}")

    with open(results_directory / "weighted_dct_shallow_water.csv", "w") as file:
        file.write("\n".join(to_write))


def weighted_sum(coefficients):
    group_0 = ((0, 0),)
    group_1 = ((1, 0), (0, 1))
    group_2 = ((2, 0), (1, 1), (0, 2))
    return (
        sum(coefficients[(...,) + coordinate].abs() for coordinate in group_0)
        + 0.5 * sum(coefficients[(...,) + coordinate].abs() for coordinate in group_1)
        + 0.25 * sum(coefficients[(...,) + coordinate].abs() for coordinate in group_2)
    )


def read_frame(data_directory, output_directory, time_step):
    with open(data_directory / output_directory / f"{time_step}.txt") as file:
        return [[float(string) for string in line.split()] for line in file.readlines()]


def pad(tensor):
    padded = torch.zeros(128, 128, dtype=tensor.dtype, device=tensor.device)
    padded[: tensor.shape[0], : tensor.shape[1]] = tensor
    return padded


if __name__ == "__main__":
    main()
