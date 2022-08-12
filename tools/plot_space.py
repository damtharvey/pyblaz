import matplotlib.pyplot as plt
import matplotlib.colors

import math


def main():
    dimensions = (2, 3, 4)
    tensor_sizes = [1 << p for p in range(2, 15)]
    float_sizes = (16, 64)
    int_sizes = (8, 32)
    block_sizes = (4, 8, 16)

    colors = list(matplotlib.colors.TABLEAU_COLORS.keys())
    horizontal_values = [str(x) for x in tensor_sizes]

    for float_size in float_sizes:
        for int_size in int_sizes:
            plt.clf()
            for line_style, n_dimensions in zip(("dotted", "dashed", "solid"), dimensions):
                for color_index, block_size in enumerate(block_sizes):
                    space_taken = [
                        space((size,) * n_dimensions, (block_size,) * n_dimensions, float_size, int_size)
                        for size in tensor_sizes
                    ]
                    plt.plot(
                        horizontal_values,
                        space_taken,
                        color=colors[color_index],
                        linestyle=line_style,
                        label=f"BS{block_size}, {n_dimensions}D",
                    )
            plt.title(f"Compressed size, float{float_size}, int{int_size}")
            plt.xlabel("array size")
            plt.xticks(rotation=-30)
            plt.ylabel("compressed memory usage (bits)")
            plt.yscale("log")
            plt.ylim((10**2.5, 10**10))
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"results/space/plots/space_float{float_size}_int{int_size}.pdf")


def space(original_shape: tuple[int, ...], block_shape: tuple[int, ...], float_size: int, int_size: int):
    n_blocks = sum(math.ceil(tensor_size / block_size) for tensor_size, block_size in zip(original_shape, block_shape))
    return (
        4
        + 2 * 64 * len(original_shape)
        + math.prod(block_shape)
        + float_size * n_blocks
        + int_size * math.prod(block_shape) * n_blocks
        + 128
    )


if __name__ == "__main__":
    main()
