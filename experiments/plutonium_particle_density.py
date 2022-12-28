import argparse
import pathlib

import torch
import numpy as np

import compression


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, default="../data/plutonium")
    parser.add_argument("--input-shape", type=str, default="40-40-66")
    args = parser.parse_args()

    input_shape = tuple(int(size) for size in args.input_shape.split("-"))
    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths = tuple((pathlib.Path(args.directory) / "n").glob("*.raw"))
    densities = torch.empty(len(paths), *input_shape)
    for index, path in enumerate(paths):
        densities[index] = torch.from_numpy(np.fromfile(path, dtype=np.uint8)).reshape(input_shape)

    densities = densities.type(dtype).to(device)

    print("uncompressed")
    print([magnitude.item() for magnitude in list((densities[1:] - densities[:-1]).norm(2, (1, 2, 3)))])

    compressor = compression.Compressor(block_shape=(8, 8, 8), dtype=dtype, index_dtype=torch.int16, device=device)

    print("compressed")
    compressed_per_time_step = [compressor.compress(densities[time_step]) for time_step in range(densities.shape[0])]

    print(
        list(
            (leading - lagging).norm_2().item()
            for (leading, lagging) in zip(compressed_per_time_step[1:], compressed_per_time_step[:-1])
        )
    )


if __name__ == "__main__":
    main()
