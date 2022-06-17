import itertools
import math

import torch
import numpy as np

import transforms
from compressed import CompressedBlock, CompressedTensor


def _test():
    dtype = torch.float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    compressor = Compressor(dtype=dtype, device=device)
    a = torch.tensor([[0.01 * x * y for y in range(1, 17)] for x in range(1, 17)], dtype=dtype, device=device)
    b = torch.tensor([[0.02 * x * y for y in range(1, 17)] for x in range(1, 17)], dtype=dtype, device=device)
    # a = torch.randn(16, 16, dtype=dtype, device=device)
    # b = torch.randn(16, 16, dtype=dtype, device=device)

    compressed_a = compressor.compress(a)
    compressed_b = compressor.compress(b)
    #
    decompressed_addition = compressor.decompress(compressed_a + compressed_b)
    addition = a + b

    # decompressed_subtraction = compressor.decompress(compressed_a - compressed_b)
    # subtraction = a - b

    decompressed_mul_constant = compressor.decompress(compressed_a * 5)
    mul_constant = a * 5
    #
    # decompressed_hadamard_product = compressor.decompress(compressed_a * compressed_b)
    # hadamard_product_a_b = a * b
    #
    decompressed_dotproduct = compressor.dot_product(compressed_a, compressed_b, 5, 6)
    dot_row_col = a[5] @ b[:, 6]

    # decompressed_matrixmul = compressor.matmultiplication(compressed_a, compressed_b)
    # matmultiplication = torch.mm(a,b)

    multiply_a_b = a @ b
    multiply_compressed_a_b = compressor.decompress(compressed_a) @ compressor.decompress(compressed_b)
    #
    print("addition error=", str((addition - decompressed_addition).norm(torch.inf)), "\n")
    # print("subtraction error=", str((subtraction - decompressed_subtraction).norm(torch.inf)), "\n")
    print(
        "multilpication with a constant error=", str((mul_constant - decompressed_mul_constant).norm(torch.inf)), "\n"
    )
    print("dot product error=", str((dot_row_col - decompressed_dotproduct).norm(torch.inf)), "\n")
    # # print("matrix multiplication error=",str((matmultiplication - decompressed_matrixmul).norm(torch.inf)),"\n")
    # print(f"hadamard product {(decompressed_hadamard_product - hadamard_product_a_b).norm(torch.inf)}")
    # print(f"matrix multiplication {(multiply_a_b - multiply_compressed_a_b).norm(torch.inf)}")


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

        self.n_coefficients = None
        self.transformer_tensor = None

        if self.n_bins <= 1 << 8:
            self.index_dtype = torch.int8
        elif self.n_bins <= 1 << 1:
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
        blocks = np.ndarray(blocks_shape, dtype=object)
        for block_index in itertools.product(*(range(size) for size in blocks_shape)):
            blocks[block_index] = self.compress_block(blocked[block_index])
        return CompressedTensor(blocks)

    def decompress(self, compressed: CompressedTensor):
        blocked = torch.empty(compressed.blocks_shape + self.block_shape, dtype=self.dtype, device=self.device)
        for block_index in itertools.product(*(range(size) for size in compressed.blocks_shape)):
            blocked[block_index] = self.decompress_block(compressed[block_index])
        return self.block_inverse(blocked)

    def compress_block(self, block: torch.Tensor) -> CompressedBlock:
        first_element, mean_slope, normalized_block = self.normalize(block)
        coefficients = self.block_transform(normalized_block)
        coefficient_indices, biggest_element = self.predict(coefficients, mean_slope)
        centered = self.center(coefficient_indices)
        return CompressedBlock(centered.type(self.index_dtype), first_element, mean_slope, biggest_element)

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

    def block(self, unblocked: torch.Tensor) -> torch.Tensor:
        """
        Section II.a

        Block Splitting

        :param unblocked: uncompressed tensor
        :return: tensor of shape blocks' shape followed by block shape.
        """
        blocked_shape = (
            *(
                (unblocked_size + block_size - 1) // block_size
                for unblocked_size, block_size in zip(unblocked.shape, self.block_shape)
            ),
            *self.block_shape,
        )
        blocked = torch.zeros(blocked_shape, dtype=self.dtype, device=self.device)
        for unblocked_indices in itertools.product(*(range(size) for size in unblocked.shape)):
            blocked[
                (
                    *(index // size for index, size in zip(unblocked_indices, self.block_shape)),
                    *(index % size for index, size in zip(unblocked_indices, self.block_shape)),
                )
            ] = unblocked[unblocked_indices]

        return blocked

    def block_inverse(self, blocked: torch.Tensor) -> torch.Tensor:
        """
        Reshape the blocked form tensor into unblocked form.

        :param blocked: tensor of shape blocks' shape followed by block shape.
        :return: unblocked tensor
        """
        unblocked_shape = (
            *(n_blocks * size for n_blocks, size in zip(blocked.shape[: self.n_dimensions], self.block_shape)),
        )
        unblocked = torch.zeros(unblocked_shape, dtype=self.dtype, device=self.device)
        for blocked_indices in itertools.product(*(range(size) for size in blocked.shape)):
            unblocked[
                (
                    *(
                        block_number * block_size + offset
                        for block_number, block_size, offset in zip(
                            blocked_indices[: self.n_dimensions],
                            self.block_shape,
                            blocked_indices[self.n_dimensions :],
                        )
                    ),
                )
            ] = blocked[blocked_indices]

        return unblocked

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
        unnormalized = (mean_slope * block).cpu()  # TODO: Consider leaving this on the GPU.
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

        if not self.n_coefficients or self.n_coefficients != n_coefficients:
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
            self.transformer_tensor = transformer_tensor

        transformed = torch.einsum(  # multiplying elements of tensor
            slope_indices.to(self.dtype),
            range(self.n_dimensions),
            self.transformer_tensor,
            range(2 * self.n_dimensions),
            range(self.n_dimensions, 2 * self.n_dimensions),
        )

        return transformed

    def dot_product(self, compressed_a: CompressedTensor, compressed_b: CompressedTensor, row: int, column: int) -> float:
        assert compressed_a.n_dimensions == 2, "Dot product not defined for dimensions other than 2."

        a_block_index = row // self.block_shape[0]
        b_block_index = column // self.block_shape[1]
        blocks_for_a = ((a_block_index, column_index) for column_index in range(compressed_a.blocks_shape[0]))
        blocks_for_b = ((row_index, b_block_index) for row_index in range(compressed_b.blocks_shape[1]))

        decompressed_dot_product = sum(
            self.dot_product_block(compressed_a[a_index], compressed_b[b_index], row % self.block_shape[0], column % self.block_shape[1])
            for a_index, b_index in zip(blocks_for_a, blocks_for_b)
        )
        return decompressed_dot_product

    def matmultiplication(self, compressed_a: CompressedTensor, compressed_b: CompressedTensor) -> torch.Tensor:
        num_blocks_rowwise_a, num_blocks_colwise_a = compressed_a.blocks_shape
        num_blocks_rowwise_b, num_blocks_colwise_b = compressed_b.blocks_shape

        numofrows = num_blocks_rowwise_a * 8
        numofcols = num_blocks_colwise_b * 8
        matproduct = torch.zeros((numofrows, numofcols), dtype=self.dtype, device=self.device)
        for row in range(0, numofrows):
            for col in range(0, numofcols):
                matproduct[row, col] = self.dot_product(compressed_a, compressed_b, row, col)
        return matproduct


if __name__ == "__main__":
    _test()
