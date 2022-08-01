import functools
import itertools
import math
import time

import torch
import torch.nn.functional
import tqdm
import numpy as np

import compressed
import transforms
from compressed import CompressedBlock, CompressedTensor


def _test():
    pass


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
        self.log_2_block_shape = tuple(size.bit_length() - 1 for size in self.block_shape)
        self.n_bins = n_bins
        self.transform = transform
        self.n_dimensions = len(block_shape)
        self.dtype = dtype
        self.device = device

        self.n_coefficients = None
        self.transformer_tensor = None
        self.inverse_transformer_tensor = None

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
        """
        This isn't blockwise complete compression. Some things are better done over the whole tensor.
        """
        assert self.n_dimensions == len(tensor.shape), (
            f"Compressor dimensionality ({self.n_dimensions}) "
            f"must match tensor dimensionality ({len(tensor.shape)})."
        )

        blocked = self.block(tensor)
        blocks_shape = blocked.shape[: self.n_dimensions]
        block_indices = tuple(itertools.product(*(range(size) for size in blocks_shape)))

        first_elements = blocked[(...,) + (0,) * self.n_dimensions]

        normalized_blocks = torch.empty(blocked.shape, dtype=self.dtype, device=self.device)

        for block_index in tqdm.tqdm(block_indices, desc="normalizing", total=math.prod(blocks_shape)):
            normalized_blocks[block_index] = self.normalize(blocked[block_index])

        coefficientss = self.blockwise_transform(normalized_blocks)
        biggest_coefficients = coefficientss.norm(torch.inf, tuple(range(self.n_dimensions, 2 * self.n_dimensions)))
        coefficientss *= (
            compressed.INDICES_RADIUS[self.index_dtype] / biggest_coefficients[(...,) + (None,) * self.n_dimensions]
        )

        return CompressedTensor(
            tensor.shape, first_elements, biggest_coefficients, coefficientss.round().type(self.index_dtype)
        )

    def decompress(self, compressed_tensor: CompressedTensor):
        """
        This isn't blockwise complete decompression. Some things are better done over the whole tensor.
        """
        assert self.n_dimensions == compressed_tensor.n_dimensions, (
            f"Compressor dimensionality ({self.n_dimensions}) "
            f"must match tensor dimensionality ({compressed_tensor.n_dimensions})."
        )

        decompressed = torch.empty(  # Don't kill
            tuple(
                ((size + block_size - 1) >> log_2_block_size) * block_size
                for size, block_size, log_2_block_size in zip(
                    compressed_tensor.original_shape, self.block_shape, self.log_2_block_shape
                )
            ),
            dtype=self.dtype,
            device=self.device,
        )

        differences = self.blockwise_transform(  # Don't kill
            (
                compressed_tensor.indicess.type(self.dtype)
                * compressed_tensor.biggest_coefficients[(...,) + (None,) * self.n_dimensions]
                / compressed.INDICES_RADIUS[self.index_dtype]
            ),
            inverse=True,
        )

        for block_index in tqdm.tqdm(
            itertools.product(*(range(size) for size in compressed_tensor.blocks_shape)),
            desc="unnormalizing",
            total=math.prod(compressed_tensor.blocks_shape),
        ):

            index_range_str = ",".join(
                f"{block_index_element * block_size} : {block_index_element * block_size + block_size}"
                for block_index_element, block_size in zip(block_index, self.block_shape)
            )
            exec(
                f"decompressed[{index_range_str}] = "
                "self.normalize_inverse(compressed_tensor.first_elements[block_index], differences[block_index])"
            )

        return eval(f"decompressed[{','.join(f':{size}' for size in compressed_tensor.original_shape)}]")

    def compress_block(self, block: torch.Tensor) -> CompressedBlock:
        normalized_block = self.normalize(block)
        coefficients = self.blockwise_transform(normalized_block[(None,) * self.n_dimensions + (...,)]).view(
            self.block_shape
        )
        biggest_coefficient = coefficients.norm(torch.inf)
        indices = self.bin(coefficients, biggest_coefficient)
        centered = self.center(indices)
        return CompressedBlock(block[(0,) * self.n_dimensions], biggest_coefficient, centered)

    def decompress_block(self, block: CompressedBlock) -> torch.Tensor:
        coefficients = self.bin_inverse(block.indices, block.biggest_coefficient)
        differences = self.blockwise_transform(coefficients[(None,) * self.n_dimensions + (...,)], inverse=True).view(
            self.block_shape
        )
        return self.normalize_inverse(block.first_element, differences)

    def dot_product_block(self, a: CompressedBlock, b: CompressedBlock, row: int, column: int) -> float:
        return self.decompress_block(a)[row] @ self.decompress_block(b)[:, column]

    def block(self, unblocked: torch.Tensor) -> torch.Tensor:
        """
        Section II.a
        Block Splitting
        :param unblocked: uncompressed tensor
        :return: tensor of shape blocks' shape followed by block shape.
        """
        padded = torch.nn.functional.pad(
            unblocked,
            list(
                itertools.chain(
                    *(
                        (0, (block_size - size) % block_size)
                        for size, block_size in zip(reversed(unblocked.shape), reversed(self.block_shape))
                    )
                )
            ),
        )

        blocks_shape = tuple(
            unblocked_size >> log_2_block_size
            for unblocked_size, log_2_block_size in zip(padded.shape, self.log_2_block_shape)
        )
        blocked_shape = blocks_shape + self.block_shape

        blocked = torch.zeros(blocked_shape, dtype=self.dtype, device=self.device)
        for block_index in tqdm.tqdm(
            itertools.product(*(range(size) for size in blocks_shape)),
            desc="blocking",
            total=math.prod(blocks_shape),
        ):
            index_range_str = ",".join(
                f"{block_index_element * block_size} : {block_index_element * block_size + block_size}"
                for block_index_element, block_size in zip(block_index, self.block_shape)
            )
            blocked[block_index] = eval(f"padded[{index_range_str}]")
        return blocked

    # def block_inverse(self, blocked: torch.Tensor) -> torch.Tensor:
    #     """
    #     Reshape the blocked form tensor into unblocked form.
    #
    #     :param blocked: tensor of shape blocks' shape followed by block shape.
    #     :return: unblocked tensor
    #     """
    #     unblocked_shape = (
    #         *(n_blocks * size for n_blocks, size in zip(blocked.shape[: self.n_dimensions], self.block_shape)),
    #     )
    #     unblocked = torch.zeros(unblocked_shape, dtype=self.dtype, device=self.device)
    #     for block_index in tqdm.tqdm(
    #             itertools.product(*(range(size) for size in blocked.shape[: self.n_dimensions])),
    #             desc="unblocking",
    #             total=math.prod(blocked.shape[: self.n_dimensions]),
    #     ):
    #         index_range_str = ",".join(
    #             f"{block_index_element * block_size} : {block_index_element * block_size + block_size}"
    #             for block_index_element, block_size in zip(block_index, self.block_shape)
    #         )
    #         exec(f"unblocked[index_range_str] = blocked[block_index]")
    #
    #     return unblocked

    def normalize(self, block: torch.Tensor) -> torch.Tensor:
        """
        Section II.b

        Block Normalization

        :param block: a block of the input
        :return: Tuple of (the first element of the block, the mean slope, the normalized block)
        """
        differences = torch.zeros_like(block, dtype=self.dtype, device=self.device)

        if self.n_dimensions == 2:  # Faster for 2D.
            differences[1:, 0] = block[1:, 0] - block[:-1, 0]
            differences[0, 1:] = block[0, 1:] - block[0, :-1]
            differences[1:, 1:] = block[1:, 1:] - (block[:-1, 1:] + block[1:, :-1]) / 2

        else:
            for n_slice_indices in range(1, self.n_dimensions + 1):
                for slice_directions in self.slice_directions_combinations(n_slice_indices):
                    assignee_index = ["1:" if index in slice_directions else "0" for index in range(self.n_dimensions)]
                    assignee_index_str = ",".join(assignee_index)

                    for direction in slice_directions:
                        shifted_index = assignee_index.copy()
                        shifted_index[direction] = ":-1"
                        shifted_index_str = ",".join(shifted_index)
                        exec(
                            f"differences[{assignee_index_str}] += "
                            f"block[{assignee_index_str}] - block[{shifted_index_str}]"
                        )

                    exec(f"differences[{assignee_index_str}] /= {n_slice_indices}")

        return differences

    def normalize_inverse(self, first_element: float, block: torch.Tensor) -> torch.Tensor:
        """
        Section inverse II.(-b)

        inverse of block normalization

        :param first_element: first element of the block before it was normalized
        :param block: a block of inverse predicted values
        :return: inverse of the normalized block
        """
        unnormalized = block.detach()
        # The first element of the normalized block should be 0.
        # We can replace it with the first element in the unnormalized tensor.
        unnormalized[(0,) * self.n_dimensions] = first_element

        if self.n_dimensions == 2:  # Faster for 2D
            unnormalized[:, 0] = torch.cumsum(block[:, 0], 0)
            unnormalized[0, :] = torch.cumsum(block[0, :], 0)
            for i in range(1, self.block_shape[0]):
                for j in range(1, self.block_shape[1]):
                    unnormalized[i, j] = block[i, j] + (unnormalized[i - 1, j] + unnormalized[i, j - 1]) / 2

        else:
            for n_slice_indices in range(1, self.n_dimensions + 1):
                for slice_directions in self.slice_directions_combinations(n_slice_indices):
                    for index_group in self.index_groups(slice_directions, unnormalized.shape):
                        assignee_indices = np.array(index_group)
                        assignee_indices_str = ",".join(
                            f"assignee_indices[:, {dimension}]" for dimension in range(self.n_dimensions)
                        )

                        for direction in slice_directions:
                            adjacent_indices = assignee_indices.copy().T
                            adjacent_indices[direction] -= 1
                            unnormalized[eval(assignee_indices_str)] += (
                                torch.take(
                                    unnormalized,
                                    torch.tensor(
                                        np.ravel_multi_index(adjacent_indices, self.block_shape), device=self.device
                                    ),
                                )
                                / n_slice_indices
                            )

        return unnormalized

    def bin(self, coefficients: torch.Tensor, biggest_coefficient: float) -> torch.Tensor:
        """
        Section II.c

        :param coefficients:
        :param biggest_coefficient:
        :return: Centered indices of the coefficient bins
        """
        return coefficients * compressed.INDICES_RADIUS[self.index_dtype] / biggest_coefficient

    def bin_inverse(self, indices: torch.Tensor, biggest_coefficient: float) -> torch.Tensor:
        """
        Section II.(-c)

        :param indices: Centered indices of the bins
        :param biggest_coefficient: biggest bin value.
        :return: Values corresponding to each index
        """
        assert biggest_coefficient >= 0, f"The biggest coefficient {biggest_coefficient} should be non-negative."

        return indices.type(self.dtype) * biggest_coefficient / compressed.INDICES_RADIUS[self.index_dtype]

    def center(self, slope_indices: torch.Tensor) -> torch.Tensor:
        """
        This adjustment follows binning and is not explicitly stated in the paper.
        We apply this adjustment trying to get closer results to the examples in the paper.

        :param slope_indices:
        :return: block of integers where the mean slope index is 0.
        """
        return slope_indices - (self.n_bins >> 1)

    def center_inverse(self, centered_slope_indices: torch.Tensor) -> torch.Tensor:
        """
        Adjust slope indices to be non-negative.

        :param centered_slope_indices:
        :return: block of non-negative slope indices
        """
        return centered_slope_indices + (self.n_bins >> 1)

    def blockwise_transform(
        self, blocked_tensor: torch.Tensor, n_coefficients: int = None, inverse=False
    ) -> torch.Tensor:
        """
        Section II.d

        Transform the blocked tensor blockwise according to self.block_shape.
        Non-hypercubic block shapes have not been tested.

        :param blocked_tensor:
        :param n_coefficients:
        :param inverse:
        :return:
        """
        if not n_coefficients:
            n_coefficients = math.prod(self.block_shape)

        if (
            (not inverse and self.transformer_tensor is None)
            or (inverse and self.inverse_transformer_tensor is None)
            or (not self.n_coefficients or self.n_coefficients != n_coefficients)
        ):
            transformer_tensor = torch.zeros(
                *self.block_shape * 2,
                dtype=self.dtype,
                device=self.device,
            )

            all_frequency_indices = sorted(
                itertools.product(*(range(size) for size in self.block_shape)),
                key=lambda x: sum(x),
            )[:n_coefficients]

            for element_indices in tqdm.tqdm(
                itertools.product(*(range(size) for size in self.block_shape)),
                desc=f"making {'inverse ' if inverse else ''}block transformer",
                total=math.prod(self.block_shape),
            ):
                for frequency_indices in all_frequency_indices:
                    transformer_tensor[(*element_indices, *frequency_indices)] = math.prod(
                        self.transform(size, element_index, frequency_index, inverse)
                        for size, element_index, frequency_index in zip(
                            self.block_shape, element_indices, frequency_indices
                        )
                    )
            if not inverse:
                self.transformer_tensor = transformer_tensor
            else:
                self.inverse_transformer_tensor = transformer_tensor
            self.n_coefficients = n_coefficients

        transformed = torch.einsum(
            blocked_tensor,
            # blocks, intrablock
            tuple(range(2 * self.n_dimensions)),
            self.transformer_tensor if not inverse else self.inverse_transformer_tensor,
            # intrablock, coefficients
            tuple(range(self.n_dimensions, 3 * self.n_dimensions)),
            # blocks, coefficients
            tuple(range(self.n_dimensions)) + tuple(range(2 * self.n_dimensions, 3 * self.n_dimensions)),
        )

        return transformed

    def dot_product(
        self, compressed_a: CompressedTensor, compressed_b: CompressedTensor, row: int, column: int
    ) -> float:
        assert (
            compressed_a.n_dimensions == 2 and compressed_b.n_dimensions == 2
        ), "Dot product not defined for dimensions other than 2."

        a_block_index = row >> self.log_2_block_shape[0]
        b_block_index = column >> self.log_2_block_shape[1]
        block_row = row % self.block_shape[0]
        block_column = column % self.block_shape[1]

        return sum(
            self.dot_product_block(compressed_a[a_index], compressed_b[b_index], block_row, block_column)
            for a_index, b_index in zip(
                ((a_block_index, column_index) for column_index in range(compressed_a.blocks_shape[0])),
                ((row_index, b_block_index) for row_index in range(compressed_b.blocks_shape[1])),
            )
        )

    def bins(self, biggest_coefficient):
        return torch.linspace(
            -biggest_coefficient - (2 * biggest_coefficient) / (self.n_bins - 2),
            biggest_coefficient,
            self.n_bins,
            dtype=self.dtype,
            device=self.device,
        )

    @functools.cache
    def slice_directions_combinations(self, n_slice_indices):
        return tuple(itertools.combinations(range(self.n_dimensions), n_slice_indices))

    @functools.cache
    def index_groups(self, slice_directions, shape):
        return tuple(
            tuple(group)
            for _, group in itertools.groupby(
                itertools.product(
                    *(range(1, size) if direction in slice_directions else (0,) for direction, size in enumerate(shape))
                ),
                key=lambda x: sum(x),
            )
        )


if __name__ == "__main__":
    _test()
