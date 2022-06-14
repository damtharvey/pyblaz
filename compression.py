from collections import namedtuple
import itertools
import math

import torch
import numpy as np

import transforms

CompressedBlock = namedtuple("CompressedBlock", ["indices", "first_element", "mean_slope", "biggest_element"])
CompressedTensor = namedtuple("CompressedTensor", ["indices", "first_elements", "mean_slopes", "biggest_elements"])


def _test():
    dtype = torch.float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    compressor = Compressor(dtype=dtype, device=device)
    # x = torch.tensor([[0.01 * x * y for y in range(8)] for x in range(8)], dtype=dtype, device=device)
    a = torch.tensor([[0.01 * x * y for y in range(8)] for x in range(8)], dtype=dtype, device=device)
    b = torch.tensor([[0.02 * x * y for y in range(8)] for x in range(8)], dtype=dtype, device=device)
    compressed_a = compressor.compress(a)
    compressed_b = compressor.compress(b)

    some_a_block = CompressedBlock(
        compressed_a.indices[0, 0],
        compressed_a.first_elements[0, 0],
        compressed_a.mean_slopes[0, 0],
        compressed_a.biggest_elements[0, 0],
    )
    some_b_block = CompressedBlock(
        compressed_b.indices[0, 0],
        compressed_b.first_elements[0, 0],
        compressed_b.mean_slopes[0, 0],
        compressed_b.biggest_elements[0, 0],
    )

    dot_6_7 = a[6] @ b[:, 7]
    print(dot_6_7)
    print(compressor.dot_product_block(some_a_block, some_b_block, 6, 7))


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
                CompressedBlock(
                    compressed.indices[block_index],
                    compressed.first_elements[block_index],
                    compressed.mean_slopes[block_index],
                    compressed.biggest_elements[block_index],
                )
            )
        return self.block_inverse(decompressed)

    def compress_block(self, block: torch.Tensor) -> tuple[torch.Tensor, float, float, float]:
        first_element, mean_slope, normalized_block = self.normalize(block)
        coefficient_indices, biggest_element = self.predict(self.block_transform(normalized_block), mean_slope)
        centered = self.center(coefficient_indices)
        stuff = centered.type(self.index_dtype), first_element, mean_slope, biggest_element
        return stuff

    def decompress_block(self, block: CompressedBlock) -> torch.Tensor:
        return self.normalize_inverse(
            block.first_element,
            block.mean_slope,
            self.block_transform(
                self.predict_inverse(self.center_inverse(block.indices), block.mean_slope, block.biggest_element),
                inverse=True,
            ),
        )

    def dot_product_block(self, a: CompressedBlock, b: CompressedBlock, row: int, column: int) -> float:
        return self.decompress_block(a)[row] @ self.decompress_block(b)[:, column]

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

        for n_slice_indices in range(1, self.n_dimensions + 1):
            for slice_directions in itertools.combinations(range(self.n_dimensions), n_slice_indices):
                assignee_index = ["1:" if index in slice_directions else "0" for index in range(self.n_dimensions)]
                assignee_index_str = ",".join(assignee_index)

                for direction in slice_directions:
                    shifted_index = assignee_index.copy()
                    shifted_index[direction] = ":-1"
                    shifted_index_str = ",".join(shifted_index)
                    exec(
                        f"differences[{assignee_index_str}] += block[{assignee_index_str}] - block[{shifted_index_str}]"
                    )

                exec(f"differences[{assignee_index_str}] /= {n_slice_indices}")

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
        unnormalized = (mean_slope * block).cpu()
        # The first element of the normalized block should be 0.
        # We can replace it with the first element in the unnormalized tensor.
        unnormalized[(0,) * self.n_dimensions] = first_element

        for n_slice_indices in range(1, self.n_dimensions + 1):
            for slice_directions in itertools.combinations(range(self.n_dimensions), n_slice_indices):
                for _, index_group in itertools.groupby(
                    itertools.product(
                        *(
                            range(1, size) if direction in slice_directions else (0,)
                            for direction, size in enumerate(unnormalized.shape)
                        )
                    ),
                    key=lambda x: sum(x),
                ):
                    assignee_indices = np.array(tuple(index_group))
                    assignee_indices_str = ",".join(
                        f"assignee_indices[:, {dimension}]" for dimension in range(self.n_dimensions)
                    )

                    for direction in slice_directions:
                        adjacent_indices = assignee_indices.copy().T
                        adjacent_indices[direction] -= 1
                        unnormalized[eval(assignee_indices_str)] += (
                            torch.take(
                                unnormalized, torch.tensor(np.ravel_multi_index(adjacent_indices, self.block_shape))
                            )
                            / n_slice_indices
                        )

        return unnormalized.to(self.device)

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

    def center(self, slope_indices: torch.Tensor) -> torch.Tensor:
        """
        This adjustment follows binning and is not explicitly stated in the paper.
        We apply this adjustment trying to get closer results to the examples in the paper.

        :param slope_indices:
        :return: block of integers where the mean slope index is 0.
        """
        return slope_indices - self.n_bins // 2 + 1

    def center_inverse(self, centered_slope_indices: torch.Tensor) -> torch.Tensor:
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

        indices = torch.zeros_like(a.indices, dtype=torch.int64)
        where_can_divide = a_indices + b_indices != 0
        indices[where_can_divide] = torch.div(
            a_indices[where_can_divide] * b_indices[where_can_divide],
            a_indices[where_can_divide] + b_indices[where_can_divide],
            rounding_mode="floor",
        )

        return CompressedBlock(
            indices.type(self.index_dtype),
            a.first_element + b.first_element,
            a.mean_slope + b.mean_slope,
            a.biggest_element + b.biggest_element,
        )

    @staticmethod
    def negate_block(block: CompressedBlock) -> CompressedBlock:
        return CompressedBlock(block.indices, -block.first_element, -block.mean_slope, block.biggest_element)

    def sub_block(self, a: CompressedBlock, b: CompressedBlock) -> CompressedBlock:
        """

        :param a: compressed block
        :param b: compressed block
        :return: the compressed sum of a and b
        """
        return self.add_block(a, self.negate_block(b))

    def const_mul_block(self, a: CompressedBlock, c: float) -> CompressedBlock:
        """

        :param a: compressed block
        :param c: scalar
        :return: the compressed sum of a and b
        """
        return CompressedBlock(a.indices, a.first_element * c, a.mean_slope * c, a.biggest_element)

    def hadamard_block(self, a: CompressedBlock, b: CompressedBlock) -> CompressedBlock:
        """

        :param a: compressed block
        :param b: compressed block
        :return: the compressed sum of a and b
        """
        a_indices = a.indices.type(torch.int64)
        b_indices = b.indices.type(torch.int64)

        first_element = a.first_element * b.first_element
        mean_slope = a.mean_slope * b.mean_slope
        biggest_element = a.biggest_element + b.biggest_element
        indices = torch.zeros_like(a.indices, dtype=torch.int64)
        where_can_divide = a_indices + b_indices != 0
        indices[where_can_divide] = torch.div(
            a_indices[where_can_divide] * b_indices[where_can_divide],
            a_indices[where_can_divide] + b_indices[where_can_divide],
            rounding_mode="floor",
        )

        return CompressedBlock(indices.type(self.index_dtype), first_element, mean_slope, biggest_element)

    def blockwise_unary(self, a: CompressedTensor, s: float, operation: callable) -> CompressedTensor:
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

    def blockwise_binary(self, a: CompressedTensor, b: CompressedTensor, operation: callable) -> CompressedTensor:
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
