import argparse
import pathlib

import torch
import numpy as np

from pyblaz import compression
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, default="./data/plutonium")
    parser.add_argument("--input-shape", type=str, default="40-40-66")
    args = parser.parse_args()
    list = [665, 670, 675, 680, 685, 686, 687, 688, 689, 690, 692, 693, 694, 695, 699]
    print("\nneutron density")
    uncompressed_n, compressed_n, decompressed_n, timesteps = print_density_temporal_difference(args, "n")
    print("\nproton density")
    uncompressed_p, compressed_p, decompressed_p, timesteps = print_density_temporal_difference(args, "p")
    print(uncompressed_n)
    print(decompressed_n)

    plt.title("L2 norm difference for each timestep  for Pu atom", fontsize=12)

    plt.plot(list[:-1], compressed_n, label="neutron-density compressed")
    plt.plot(list[:-1], decompressed_n, label="neutron-density (de)compressed", linestyle="-.")
    plt.plot(list[:-1], uncompressed_n, label="neutron-density uncompressed", linestyle="dotted")

    # plt.plot(list[:-1], compressed_p, label="proton-density compressed")
    # plt.plot(list[:-1], uncompressed_p, label="proton-density uncompressed", linestyle="dotted")

    plt.legend(fontsize=11, loc="upper left")
    plt.xlabel("Time Steps", fontsize=12)
    plt.ylabel("L2 norm value", fontsize=12)
    plt.xticks(fontsize=12)

    plt.savefig("L2norm_Pu.png")
    plt.show()
    plt.close()
    diff1 = []
    diff2 = []
    diff3 = []
    for i in range(len(uncompressed_n)):
        diff1.append(abs(uncompressed_n[i] - decompressed_n[i]))
        diff2.append(abs(uncompressed_n[i] - compressed_n[i]))
        diff3.append(abs(compressed_n[i] - decompressed_n[i]))

    print(max(diff1))
    print(max(diff2))
    print(max(diff3))

    # print(print_wasserstein_distance(args, "n"))


# def softmax(x):
#     """Compute softmax values for each sets of scores in x."""
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum()


# def print_wasserstein_distance(args, particle_name: str):
#     input_shape = tuple(int(size) for size in args.input_shape.split("-"))
#     dtype = torch.float32
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     paths = tuple((pathlib.Path(args.directory) / particle_name).glob("*.raw"))

#     densities = torch.empty(len(paths), *input_shape)
#     for index, path in enumerate(paths):
#         densities[index] = torch.from_numpy(np.fromfile(path, dtype=np.uint8)).reshape(input_shape)
#     densities = densities.type(dtype).to(device)

#     order = 68
#     uncompressed_densities = []
#     print("uncompressed")
#     for i in range(len(densities)):
#         uncompressed_densities.append(np.sort(softmax(np.asarray(densities[i])), axis=None))
#     print(len(uncompressed_densities))
#     uncompressed_list_compressed_wass = []
#     uncompressed_per_time_step = []
#     for i in range(len(uncompressed_densities) - 1):
#         uncompressed_per_time_step = [
#             ((abs(uncompressed_densities[i][a] - uncompressed_densities[i + 1][b])) ** order).mean() ** (1 / order)
#             for a in range(len(uncompressed_densities[i]))
#             for b in range(len(uncompressed_densities[i + 1]))
#         ]
#     uncompressed_list_compressed_wass.append(
#         np.mean(uncompressed_per_time_step),
#     )
#     return uncompressed_per_time_step


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
    print(len(uncompressed_result))
    compressor = compression.PyBlaz(block_shape=(16, 16, 16), dtype=dtype, index_dtype=torch.int16, device=device)
    print("compressed")
    timesteps = [timestep for timestep in range(len(densities) - 1)]
    compressed_per_time_step = [compressor.compress(densities[time_step]) for time_step in range(densities.shape[0])]
    decompressed_per_time_step = [
        compressor.decompress(compressor.compress(densities[time_step])) for time_step in range(densities.shape[0])
    ]
    print(
        list(
            (leading - lagging).norm(2)
            for (leading, lagging) in zip(decompressed_per_time_step[1:], decompressed_per_time_step[:-1])
        )
    )
    decompressed_result = list(
        (leading - lagging).norm(2)
        for (leading, lagging) in zip(decompressed_per_time_step[1:], decompressed_per_time_step[:-1])
    )

    compressed_result = list(
        (leading - lagging).norm_2().item()
        for (leading, lagging) in zip(compressed_per_time_step[1:], compressed_per_time_step[:-1])
    )
    print(len(compressed_result))
    return uncompressed_result, compressed_result, decompressed_result, timesteps


if __name__ == "__main__":
    main()
