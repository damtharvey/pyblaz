import pathlib
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors


def main():
    colors = list(matplotlib.colors.TABLEAU_COLORS.keys())

    tuples = []
    with open("results/space/blaz_file_sizes.txt") as file:
        for line in file.readlines():
            split = line.split()
            matrix_size = int(split[8][22:-5])
            file_size = int(split[4]) * 8  # given in bytes, converting to bits
            tuples.append((matrix_size, file_size))
    tuples = sorted(tuples)

    matrix_sizes = [x[0] for x in tuples]
    horizontal_values = [str(x) for x in matrix_sizes]
    compressed_sizes = np.array([x[1] for x in tuples])
    theoretical_compressed_sizes = [theoretical_compressed_size(size) for size in matrix_sizes]
    uncompressed_sizes = np.array([uncompressed_size(size, 64) for size in matrix_sizes])
    memory_savings = uncompressed_sizes - compressed_sizes
    expected_memory_savings = uncompressed_sizes - theoretical_compressed_sizes
    compression_ratio = uncompressed_sizes / compressed_sizes
    expected_compression_ratio = uncompressed_sizes / theoretical_compressed_sizes
    plt.clf()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.title(f"Blaz memory usage")
    ax1.set_xlabel("matrix size")
    ax1.set_xticklabels(horizontal_values, rotation=-30)
    ax1.set_ylabel("compressed size (bits)")
    ax1.set_yscale("log")
    ax2.set_ylabel("compression ratio", color=colors[2])
    ax1.plot(horizontal_values, compressed_sizes, color=colors[0], label="compressed size")
    ax1.plot(horizontal_values, theoretical_compressed_sizes, color=colors[0], linestyle="dashed")
    ax1.plot(horizontal_values, memory_savings, color=colors[1], label="memory savings")
    ax1.plot(horizontal_values, expected_memory_savings, color=colors[1], linestyle="dashed")
    ax2.plot(horizontal_values, compression_ratio, color=colors[2], label="compression ratio")
    ax2.plot(
        horizontal_values, expected_compression_ratio, color=colors[2], linestyle="dashed", label="compression ratio"
    )
    ax1.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"results/space/plots/blaz_space_ratio.pdf")


def uncompressed_size(original_size: int, float_size: int):
    return 128 + float_size * original_size * original_size


def theoretical_compressed_size(original_size):
    n_blocks = math.ceil(original_size / 8) ** 2
    return 32 + 32 + 2 * n_blocks * 64 + n_blocks * 28 * 8


if __name__ == "__main__":
    main()
