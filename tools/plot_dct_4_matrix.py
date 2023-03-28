import math
import pathlib

import torch
import numpy as np

import matplotlib.pyplot as plt


def main():
    save_path = pathlib.Path("helper_figures")
    save_path.mkdir(parents=True, exist_ok=True)

    block_size = 4
    dtype = torch.float64

    transform_matrix = np.array(
        [
            [cosine(block_size, element, frequency, False) for frequency in range(block_size)]
            for element in range(block_size)
        ]
    )
    plt.imshow(transform_matrix, cmap="gray")
    plt.axis("off")
    plt.savefig(save_path / "4x4_transform_matrix.pdf")
    print(transform_matrix @ np.array([5, 9, 2, 6]))


def cosine(block_size: int, element: int, frequency: int, inverse: bool = False) -> float:
    if inverse:
        element, frequency = frequency, element
    return math.sqrt((1 + (frequency > 0)) / block_size) * math.cos(
        (2 * element + 1) * frequency * math.pi / (2 * block_size)
    )


if __name__ == "__main__":
    main()
