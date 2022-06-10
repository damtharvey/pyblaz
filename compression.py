from collections import namedtuple
import itertools
import math
from typing import overload

import torch

import transforms

CompressedBlock = namedtuple("CompressedBlock", ["indices", "first_element", "mean_slope", "biggest_element"])
CompressedTensor = namedtuple("CompressedTensor", ["indices", "first_elements", "mean_slopes", "biggest_elements"])


def _test():
    import tqdm

    dtype = torch.float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    compressor = Compressor(dtype=dtype, device=device)
    # x = torch.tensor([[0.01 * x * y for y in range(8)] for x in range(8)], dtype=dtype, device=device)
    a = torch.tensor([[0.01 * x * y for y in range(8)] for x in range(8)], dtype=dtype, device=device)
    b = torch.tensor([[0.02 * x * y for y in range(8)] for x in range(8)], dtype=dtype, device=device)
    c_add = a + b
    c_sub = a - b
    compressed_a = compressor.compress(a)
    compressed_b = compressor.compress(b)

    compressed_c_maybe_add = compressor.blockwise(compressed_a, compressed_b, compressor.add_block)

    c_hat_add = compressor.decompress(compressed_c_maybe_add)
    print("\n\n\n******************Addition****************************")
    print(c_add)
    print(c_hat_add)
    print((c_add - c_hat_add).norm(torch.inf))

    compressed_c_maybe_sub = compressor.blockwise(compressed_a, compressed_b, compressor.sub_block)

    c_hat_sub = compressor.decompress(compressed_c_maybe_sub)
    print("\n\n\n******************Subtraction************************")
    print(c_sub)
    print(c_hat_sub)
    print((c_sub - c_hat_sub).norm(torch.inf))


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

        if self.n_bins <= 1 << 8:
            self.index_dtype = torch.int8
        elif self.n_bins <= 1 << 16:
            self.index_dtype = torch.int16
        elif self.n_bins <= 1 << 32:
            self.index_dtype = torch.int32
        elif self.n_bins <= 1 << 64:
            self.index_dtype = torch.int64
        else:
            raise ValueError("Too many bins.")

    def compress(self, tensor: torch.Tensor) -> CompressedTensor:
        blocked = self.block(tensor)
        blocks_shape = blocked.shape[: self.n_dimensions]
        indices = torch.zeros(blocked.shape, dtype=self.index_dtype, device=self.device)
        first_elements = torch.zeros(blocks_shape, dtype=self.dtype, device=self.device)
        mean_slopes = torch.zeros(blocks_shape, dtype=self.dtype, device=self.device)
        biggest_elements = torch.zeros(blocks_shape, dtype=self.dtype, device=self.device)

        for block_index in itertools.product(*(range(size) for size in blocks_shape)):
            (
                indices[block_index],
                first_elements[block_index],
                mean_slopes[block_index],
                biggest_elements[block_index],
            ) = self.compress_block(blocked[block_index])

        return CompressedTensor(indices, first_elements, mean_slopes, biggest_elements)

    def decompress(
        self,
        compressed: CompressedTensor,
    ):
        decompressed = torch.zeros(compressed.indices.shape, dtype=self.dtype, device=self.device)
        blocks_shape = decompressed.shape[: self.n_dimensions]
        for block_index in itertools.product(*(range(size) for size in blocks_shape)):
            decompressed[block_index] = self.decompress_block(
                compressed.indices[block_index],
                compressed.first_elements[block_index],
                compressed.mean_slopes[block_index],
                compressed.biggest_elements[block_index],
            )
        return self.block_inverse(decompressed)

    def compress_block(self, block) -> tuple[torch.Tensor, float, float, float]:
        first_element, mean_slope, normalized_block = self.normalize(block)
        coefficient_indices, biggest_element = self.predict(self.block_transform(normalized_block), mean_slope)
        centered = self.center(coefficient_indices)
        stuff = centered.type(self.index_dtype), first_element, mean_slope, biggest_element
        return stuff

    def decompress_block(
        self, indices: torch.Tensor, first_element: float, mean_slope: float, biggest_element: float
    ) -> torch.Tensor:
        return self.normalize_inverse(
            first_element,
            mean_slope,
            self.block_transform(
                self.predict_inverse(self.center_inverse(indices), mean_slope, biggest_element), inverse=True
            ),
        )

    def block(self, preimage: torch.Tensor) -> torch.Tensor:
        """
        Section II.a

        Block Splitting

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
        # TODO consider using view.
        return torch.reshape(preimage, image_shape)

    def block_inverse(self, blocked: torch.Tensor) -> torch.Tensor:
        """
        Reshape the blocked form tensor into unblocked form.

        :param blocked: tensor of shape blocks' shape followed by block shape.
        :return: unblocked tensor
        """
        return torch.reshape(
            blocked,
            tuple(n_blocks * size for n_blocks, size in zip(blocked.shape[: self.n_dimensions], self.block_shape)),
        )

    def normalize(self, block: torch.Tensor) -> tuple[torch.float, torch.float, torch.Tensor]:
        """
        Section II.b

        Block Normalization

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

        # TODO This is still 2D.
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

    def normalize_inverse(self, first_element: float, mean_slope: float, block: torch.Tensor) -> torch.Tensor:
        """
        Section inverse II.(-b)

        inverse of block normalization

        :param first_element: first element of the block before it was normalized
        :param mean_slope: mean slope of the differences
        :param block: a block of inverse predicted values
        :return: inverse of the normalized block
        """

        # TODO don't be hacky
        new_array = torch.zeros_like(block)
        block_array = mean_slope * block
        row, col = self.block_shape
        for x in range(row):
            for y in range(col):
                if x == 0 and y == 0:
                    new_array[x][y] = first_element
                elif x == 0 and y != 0:
                    new_array[x][y] = block_array[x][y] + new_array[x][y - 1]
                elif x != 0 and y == 0:
                    new_array[x][y] = block_array[x][y] + new_array[x - 1][y]
                else:
                    new_array[x][y] = (
                        (block_array[x][y] + new_array[x - 1][y]) + (block_array[x][y] + new_array[x][y - 1])
                    ) / 2

        return new_array

    @staticmethod
    def mean_slope(block: torch.Tensor) -> torch.float:
        return block.sum() / torch.count_nonzero(block)

    def predict(self, normalized_block: torch.Tensor, mean_slope: torch.float) -> tuple[torch.Tensor, torch.float]:
        """
        Section II.c

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
        return (normalized_block.unsqueeze(-1) - slopes).abs().min(-1).indices, biggest_element

    def predict_inverse(
        self, indices: torch.Tensor, mean_slope: torch.float, biggest_element: torch.float
    ) -> torch.Tensor:
        """
        Section II.(-c)

        :param indices: Indices of the bins
        :param mean_slope: mean slope of the differences
        :param biggest_element: biggest element of the normalized block before binning
        :return: Values corresponding to each index
        """
        slopes = torch.linspace(
            mean_slope - biggest_element,
            mean_slope + biggest_element,
            self.n_bins,
            dtype=self.dtype,
            device=self.device,
        )
        return slopes[indices.type(torch.int64)]

    def center(self, slope_indices):
        """
        This adjustment follows binning and is not explicitly stated in the paper.
        We apply this adjustment trying to get closer results to the examples in the paper.

        :param slope_indices:
        :return: block of integers where the mean slope index is 0.
        """
        return slope_indices - self.n_bins // 2 + 1

    def center_inverse(self, centered_slope_indices):
        """
        Adjust slope indices to be non-negative.

        :param centered_slope_indices:
        :return: block of non-negative slope indices
        """
        return centered_slope_indices + self.n_bins // 2 - 1

    def block_transform(self, slope_indices: torch.Tensor, n_coefficients: int = None, inverse=False) -> torch.Tensor:
        """
        Section II.d

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

        transformed = torch.einsum(  # multiplying elements of tensor
            slope_indices.to(self.dtype),
            range(self.n_dimensions),
            transformer_tensor,
            range(2 * self.n_dimensions),
            range(self.n_dimensions, 2 * self.n_dimensions),
        )

        return transformed

    def add_block(self, a: CompressedBlock, b: CompressedBlock) -> CompressedBlock:
        """

        :param a: compressed block
        :param b: compressed block
        :return: the compressed sum of a and b
        """
        a_indices = a.indices.type(torch.int64)
        b_indices = b.indices.type(torch.int64)

        first_element = a.first_element + b.first_element
        mean_slope = a.mean_slope + b.mean_slope
        biggest_element = a.biggest_element + b.biggest_element
        indices = torch.zeros_like(a.indices, dtype=torch.int64)
        for row_index in range(a.indices.shape[0]):
            for column_index in range(a.indices.shape[1]):
                if a.indices[row_index, column_index] + b.indices[row_index, column_index] == 0:
                    indices[row_index, column_index] = 0
                else:
                    indices[row_index, column_index] = torch.div(
                        a_indices[row_index, column_index] * b_indices[row_index, column_index],
                        a_indices[row_index, column_index] + b_indices[row_index, column_index],
                        rounding_mode="floor",
                    )

        return CompressedBlock(indices.type(self.index_dtype), first_element, mean_slope, biggest_element)

    def sub_block(self, a: CompressedBlock, b: CompressedBlock) -> CompressedBlock:
        """

        :param a: compressed block
        :param b: compressed block
        :return: the compressed sum of a and b
        """
        a_indices = a.indices.type(torch.int64)
        b_indices = b.indices.type(torch.int64)

        first_element = a.first_element - b.first_element
        mean_slope = a.mean_slope - b.mean_slope
        biggest_element = a.biggest_element + b.biggest_element
        indices = torch.zeros_like(a.indices, dtype=torch.int64)
        for row_index in range(a.indices.shape[0]):
            for column_index in range(a.indices.shape[1]):
                if a.indices[row_index, column_index] + b.indices[row_index, column_index] == 0:
                    indices[row_index, column_index] = 0
                else:
                    indices[row_index, column_index] = torch.div(
                        a_indices[row_index, column_index] * b_indices[row_index, column_index],
                        a_indices[row_index, column_index] + b_indices[row_index, column_index],
                        rounding_mode="floor",
                    )

        return CompressedBlock(indices.type(self.index_dtype), first_element, mean_slope, biggest_element)

    @overload
    def blockwise(self, a: CompressedTensor, s: float, operation: callable) -> CompressedTensor:
        blocks_shape = a.indices.shape[: self.n_dimensions]
        indices = torch.zeros(a.indices.shape, dtype=self.index_dtype, device=self.device)
        first_elements = torch.zeros(blocks_shape, dtype=self.dtype, device=self.device)
        mean_slopes = torch.zeros(blocks_shape, dtype=self.dtype, device=self.device)
        biggest_elements = torch.zeros(blocks_shape, dtype=self.dtype, device=self.device)

        for block_index in itertools.product(*(range(size) for size in blocks_shape)):
            (
                indices[block_index],
                first_elements[block_index],
                mean_slopes[block_index],
                biggest_elements[block_index],
            ) = operation(
                CompressedBlock(
                    a.indices[block_index],
                    a.first_elements[block_index],
                    a.mean_slopes[block_index],
                    a.biggest_elements[block_index],
                ),
                s,
            )

        return CompressedTensor(indices, first_elements, mean_slopes, biggest_elements)

    def blockwise(self, a: CompressedTensor, b: CompressedTensor, operation: callable) -> CompressedTensor:
        blocks_shape = a.indices.shape[: self.n_dimensions]
        indices = torch.zeros(a.indices.shape, dtype=self.index_dtype, device=self.device)
        first_elements = torch.zeros(blocks_shape, dtype=self.dtype, device=self.device)
        mean_slopes = torch.zeros(blocks_shape, dtype=self.dtype, device=self.device)
        biggest_elements = torch.zeros(blocks_shape, dtype=self.dtype, device=self.device)

        for block_index in itertools.product(*(range(size) for size in blocks_shape)):
            (
                indices[block_index],
                first_elements[block_index],
                mean_slopes[block_index],
                biggest_elements[block_index],
            ) = operation(
                CompressedBlock(
                    a.indices[block_index],
                    a.first_elements[block_index],
                    a.mean_slopes[block_index],
                    a.biggest_elements[block_index],
                ),
                CompressedBlock(
                    b.indices[block_index],
                    b.first_elements[block_index],
                    b.mean_slopes[block_index],
                    b.biggest_elements[block_index],
                ),
            )

        return CompressedTensor(indices, first_elements, mean_slopes, biggest_elements)


if __name__ == "__main__":
    _test()