import itertools
import math
import pathlib
import string
from typing import Callable, Tuple, Union
import tqdm

import torch

import transforms


def _test():
    dtype = torch.float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    compressor = Compressor(dtype=dtype, device=device)
    x = torch.tensor([[0.01 * x * y for y in range(8)] for x in range(8)], dtype=dtype, device=device)
    blocked = compressor.block(x)
    first_element, mean_slope, normalized_block = compressor.normalize(blocked[0, 0])
    print(normalized_block)

    # slope_indices = compressor.predict(normalized_block, mean_slope)
    # test_case_slope_indices = torch.tensor(
    #     [
    #         [125, 125, 125, 125, 125, 125, 125, 125],
    #         [125, 143, 153, 162, 171, 180, 189, 198],
    #         [125, 153, 162, 171, 180, 189, 198, 207],
    #         [125, 162, 171, 180, 189, 198, 207, 217],
    #         [125, 171, 180, 189, 198, 207, 217, 226],
    #         [125, 180, 189, 198, 207, 217, 226, 235],
    #         [125, 189, 198, 207, 217, 226, 235, 244],
    #         [125, 198, 207, 217, 226, 235, 244, 253],
    #     ],
    #     dtype=dtype,
    #     device=device,
    # )
    # scale, coefficients = compressor.block_transform(test_case_slope_indices)
    # print(scale)
    # print(coefficients)


class Compressor:
    """
    blaz compressor as in https://arxiv.org/abs/2202.13007
    """

    def __init__(
        self,
        block_shape: tuple[int, ...] = (8, 8),
        n_bins: int = 256,
        transform: callable = transforms.cosine,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cuda"),
    ):
        self.block_shape = block_shape
        self.n_bins = n_bins
        self.transform = transform
        self.n_dimensions = len(block_shape)
        self.dtype = dtype
        self.device = device

        self.block_first_elements = None
        self.block_mean_slopes = None

    def block(self, preimage: torch.Tensor) -> torch.Tensor:
        """
        Section IIa

        :param preimage: uncompressed tensor
        :return: tensor of shape blocks' shape followed by block shape.
        """
        image_shape = (
            *(
                (preimage_size + block_size - 1) // block_size
                for preimage_size, block_size in zip(preimage.shape, self.block_shape)
            ),
            *self.block_shape,
        )
        return torch.reshape(preimage, image_shape)

    def normalize(self, block: torch.Tensor) -> tuple[torch.float, torch.float, torch.Tensor]:
        """
        Section IIb

        :param block: a block of the input
        :return: Tuple of (the first element of the block, the mean slope, the normalized block)
        """
        differences = torch.zeros_like(block, dtype=self.dtype, device=self.device)
        for dimension in range(self.n_dimensions):
            exec(
                f"differences[{'1:,' * dimension} 0, {'1:,' * (self.n_dimensions - dimension - 1)}] "
                f"= block[{'1:,' * dimension} 0, {'1:,' * (self.n_dimensions - dimension - 1)}] "
                f"- block[{':-1,' * dimension} 0, {':-1,' * (self.n_dimensions - dimension - 1)}]"
            )

        inner_string = "".join(
            f" - block[{'1:,' * dimension} :-1, {'1:,' * (self.n_dimensions - dimension - 1)}]"
            for dimension in range(self.n_dimensions)
        )
        exec(
            f"differences[{'1:,' * self.n_dimensions}] "
            f"= ({self.n_dimensions} * block[{'1:,' * self.n_dimensions}]{inner_string}) / {self.n_dimensions}"
        )

        mean_slope = self.mean_slope(differences)
        return block[(0,) * self.n_dimensions], mean_slope, differences / mean_slope

    @staticmethod
    def mean_slope(block: torch.Tensor) -> torch.float:
        return block.sum() / torch.count_nonzero(block)

    def predict(self, normalized_block: torch.Tensor, mean_slope: torch.float) -> torch.Tensor:
        """
        Section IIc

        :param normalized_block:
        :param mean_slope:
        :return: Indices of the slope bins
        """
        biggest_element = normalized_block.norm(torch.inf)
        slopes = torch.linspace(
            mean_slope - biggest_element,
            mean_slope + biggest_element,
            self.n_bins,
            dtype=self.dtype,
            device=self.device,
        )
        return (normalized_block.unsqueeze(-1) - slopes).abs().min(-1).indices

    def block_transform(
        self, slope_indices: torch.Tensor, n_coefficients: int = None, inverse=False
    ) -> tuple[torch.float, torch.Tensor]:
        """
        Section IId

        :param slope_indices:
        :param n_coefficients:
        :param inverse:
        :return:
        """
        if not n_coefficients:
            n_coefficients = math.prod(self.block_shape)

        transformer_tensor = torch.zeros(
            *self.block_shape * 2,
            dtype=self.dtype,
            device=self.device,
        )

        all_frequency_indices = sorted(
            itertools.product(*(range(size) for size in self.block_shape)),
            key=lambda x: sum(x),
        )[:n_coefficients]

        for element_indices in itertools.product(*(range(size) for size in self.block_shape)):
            for frequency_indices in all_frequency_indices:
                transformer_tensor[(*element_indices, *frequency_indices)] = math.prod(
                    self.transform(size, element_index, frequency_index, inverse)
                    for size, element_index, frequency_index in zip(
                        self.block_shape, element_indices, frequency_indices
                    )
                )

        transformed = torch.einsum(
            slope_indices.to(self.dtype)
            / slope_indices.abs().max(),  # Not in paper, but need it to get scale close to 20.
            range(self.n_dimensions),
            transformer_tensor,
            range(2 * self.n_dimensions),
            range(self.n_dimensions, 2 * self.n_dimensions),
        )

        scale = 127 / transformed.norm(torch.inf)
        return scale, (transformed * scale).to(torch.long)


if __name__ == "__main__":
    _test()
