import argparse
import pathlib

import torch
import numpy as np

import compression
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, default="../data/plutonium")
    parser.add_argument("--input-shape", type=str, default="40-40-66")
    args = parser.parse_args()

    print("\nneutron density")
    print_density_temporal_difference(args, "n")
    print("\nproton density")
    print_density_temporal_difference(args, "p")


def print_density_temporal_difference(args, particle_name: str):
    input_shape = tuple(int(size) for size in args.input_shape.split("-"))
    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = tuple((pathlib.Path(args.directory) / particle_name).glob("*.raw"))

    densities = torch.empty(len(paths), *input_shape)
    for index, path in enumerate(paths):
        densities[index] = torch.from_numpy(np.fromfile(path, dtype=np.uint8)).reshape(input_shape)
    densities = densities.type(dtype).to(device)

    print("uncompressed")
    uncompressed_result = [magnitude.item() for magnitude in list((densities[1:] - densities[:-1]).norm(2, (1, 2, 3)))]
    print(uncompressed_result)
    compressor = compression.Compressor(block_shape=(8, 8, 8), dtype=dtype, index_dtype=torch.int8, device=device)
    print("compressed")
    timesteps = [timestep for timestep in range(len(densities) - 1)]
    compressed_per_time_step = [compressor.compress(densities[time_step]) for time_step in range(densities.shape[0])]

    compressed_result = list(
        (leading - lagging).norm_2().item()
        for (leading, lagging) in zip(compressed_per_time_step[1:], compressed_per_time_step[:-1])
    )
    print(compressed_result)

    plt.xlabel("timestep")
    plt.ylabel("L2 norm value")
    plt.title("L2 norm difference for each timestep for" + particle_name + "for Pu atom")
    plt.xticks(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        [
            "665",
            "670",
            "675",
            "680",
            "686",
            "687",
            "688",
            "689",
            "690",
            "692",
            "693",
            "694",
            "695",
            "699",
        ],
    )
    plt.plot(np.asarray(timesteps), np.asarray(uncompressed_result), label="uncompressed data")

    plt.xticks(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        [
            "665",
            "670",
            "675",
            "680",
            "686",
            "687",
            "688",
            "689",
            "690",
            "692",
            "693",
            "694",
            "695",
            "699",
        ],
    )

    plt.plot(np.asarray(timesteps), np.asarray(compressed_result), label="(de)compressed data")
    plt.legend()
    plt.savefig("L2norm_" + particle_name + ".png")
    plt.close()


if __name__ == "__main__":
    main()
