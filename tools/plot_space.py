import argparse

import math
import pathlib
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", action="store_true")
    args = parser.parse_args()

    colors = list(matplotlib.colors.TABLEAU_COLORS.keys())

    # like_shallow_water()

    sweep_settings(colors, args.ratio)


def like_shallow_water():
    dimensions = 2
    tensor_size = (79, 55)
    float_size= 64
    int_size = 8
    block_size = 8
    n_time_steps = 10**6
    horizontal_values = []
    uncompressed_sizes = []
    compressed_sizes = []
    for time_step in range(1, n_time_steps + 1, n_time_steps // 10):
        horizontal_values.append(time_step)
        uncompressed_sizes.append(time_step * uncompressed_size(tensor_size, float_size))
        compressed_sizes.append(
            time_step * compressed_size(tensor_size, (block_size,) * dimensions, 1, float_size, int_size)
        )
    uncompressed_sizes = np.array(uncompressed_sizes)
    compressed_sizes = np.array(compressed_sizes)
    memory_savings = uncompressed_sizes - compressed_sizes
    relative_size = compressed_sizes / uncompressed_sizes
    plt.clf()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.title(f"Memory savings")
    ax1.set_xlabel("time step")
    ax1.set_ylabel("memory saving")
    ax2.set_ylabel("compressed relative size")
    ax1.plot(horizontal_values, memory_savings)
    ax2.plot(horizontal_values, relative_size, label="learning rate")
    ax1.legend()
    plt.tight_layout()
    # plt.savefig(f"results/space/plots/space_shallow_water.pdf")
    plt.show()


def sweep_settings(colors, ratio=False):
    save_path = pathlib.Path("results/space/plots")
    save_path.mkdir(parents=True, exist_ok=True)

    dimensions = (2, 3, 4)
    tensor_sizes = [1 << p for p in range(2, 15)]
    float_sizes = (16, 64)
    int_sizes = (8, 32)
    # block_sizes = (4, 8, 16)
    block_size = 8
    keep_rates = (1, 0.5)
    horizontal_values = [str(x) for x in tensor_sizes]
    for float_size in float_sizes:
        for int_size in int_sizes:
            plt.clf()
            for line_style, n_dimensions in zip(reversed(("dotted", "dashed", "solid")), reversed(dimensions)):
                # for color_index, block_size in enumerate(block_sizes):
                    # keep_rate = 1
                for color_index, keep_rate in enumerate(keep_rates):
                    if ratio:
                        compression_ratio = [
                            uncompressed_size((tensor_size,) * n_dimensions)
                            / compressed_size(
                                (tensor_size,) * n_dimensions, (block_size,) * n_dimensions, keep_rate, float_size, int_size
                            )
                            for tensor_size in tensor_sizes
                        ]
                        plt.plot(
                            horizontal_values,
                            compression_ratio,
                            color=colors[color_index],
                            linestyle=line_style,
                            label=f"{n_dimensions}D, keep {int(keep_rate * 100)}%",
                            # label=f"{n_dimensions}D, BS{block_size}",
                        )
                    else:
                        space_taken = [
                            compressed_size((size,) * n_dimensions, (block_size,) * n_dimensions, keep_rate, float_size, int_size)
                            for size in tensor_sizes
                        ]
                        plt.plot(
                            horizontal_values,
                            space_taken,
                            color=colors[color_index],
                            linestyle=line_style,
                            label=f"{n_dimensions}D, keep {int(keep_rate * 100)}%",
                            # label=f"{n_dimensions}D, BS{block_size}",
                        )
            if ratio:
                plt.title(f"Compression ratio using float{float_size} and int{int_size}")
                plt.xlabel("array size")
                plt.xticks(rotation=-30)
                plt.ylabel("uncompressed size / compressed size")
                # plt.yscale("log")
                plt.legend()
                plt.tight_layout()
                plt.savefig(save_path / f"ratio_float{float_size}_int{int_size}.pdf")
            else:
                plt.title(f"Compressed-array memory usage using float{float_size} and int{int_size}")
                plt.xlabel("array size")
                plt.xticks(rotation=-30)
                plt.ylabel("compressed-array memory usage (bits)")
                plt.yscale("log")
                # plt.ylim((10**2.5, 10**10))
                plt.legend()
                plt.tight_layout()
                plt.savefig(save_path / f"space_float{float_size}_int{int_size}.pdf")


def compressed_size(original_shape: tuple[int, ...], block_shape: tuple[int, ...], keep_proportion: float, float_size: int, int_size: int):
    n_blocks = math.prod(math.ceil(tensor_size / block_size) for tensor_size, block_size in zip(original_shape, block_shape))
    return (
        4
        + 64 * len(original_shape)
        + 64
        + 64 * len(original_shape)
        + math.prod(block_shape)
        + float_size * n_blocks
        + int_size * keep_proportion * n_blocks * math.prod(block_shape)
    )


def uncompressed_size(original_shape: tuple[int, ...]):
    return 64 * len(original_shape) + 64 + 64 * math.prod(original_shape)


if __name__ == "__main__":
    main()
