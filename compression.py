import itertools
import math
import time

import torch
import tqdm
import numpy as np

import transforms
from compressed import CompressedBlock, CompressedTensor


def _test():
    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    compressor = Compressor(block_shape=(8, 8), n_bins=256, dtype=dtype, device=device)

    x = torch.randn(16, 16, dtype=dtype, device=device) * 10
    compressed_x = compressor.compress(x)

    # decompressed_x = compressor.decompress(compressed_x)
    #
    # print((x - decompressed_x).norm(torch.inf))
    # print(
    #     f"{compressor.bin_inverse_time=}\n"
    #     f"{compressor.block_transform_inverse_time=}\n"
    #     f"{compressor.normalize_inverse_time=}"
    # )

    # compressed_negated_x = -compressed_x
    # print((x + compressor.decompress(compressed_negated_x)).norm(torch.inf))

    x = torch.tensor([[0.1 * i * j for j in range(1, 17)] for i in range(1, 17)], dtype=dtype, device=device)
    compressed_x = compressor.compress(x)
    y = torch.tensor([[0.2 * i * j for j in range(1, 17)] for i in range(1, 17)], dtype=dtype, device=device)
    compressed_y = compressor.compress(x)
    compressed_sum = compressed_x + compressed_y
    decompressed_sum = compressor.decompress(compressed_sum)
    print(((x + y) - decompressed_sum).norm(torch.inf))


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
        self.bin_inverse_time = 0
        self.block_transform_inverse_time = 0
        self.normalize_inverse_time = 0

        self.block_shape = block_shape
        self.n_bins = n_bins
        self.transform = transform
        self.n_dimensions = len(block_shape)
        self.dtype = dtype
        self.device = device

        self.index_groups = None

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
        blocks_shape = (
            *(
                (unblocked_size + block_size - 1) // block_size
                for unblocked_size, block_size in zip(tensor.shape, self.block_shape)
            ),
        )
        first_elements = torch.empty(blocks_shape, dtype=self.dtype, device=self.device)
        biggest_coefficients = torch.empty(blocks_shape, dtype=self.dtype, device=self.device)
        indicess = torch.empty(blocks_shape + self.block_shape, dtype=self.index_dtype, device=self.device)

        for block_index in tqdm.tqdm(
            itertools.product(*(range(size) for size in blocks_shape)),
            desc="blockwise compression",
            total=math.prod(blocks_shape),
        ):
            index_range_str = ",".join(
                f"{block_index_element * block_size} : {block_index_element * block_size + block_size}"
                for block_index_element, block_size in zip(block_index, self.block_shape)
            )
            first_elements[block_index] = tensor[tuple(torch.tensor(block_index) * torch.tensor(self.block_shape))]
            # biggest_coefficient, indices = self.compress_block(eval(f"tensor[{index_range_str}]"))
            # biggest_coefficients[block_index] = biggest_coefficient
            # indicess[block_index] = indices

            normalized_block = self.normalize(eval(f"tensor[{index_range_str}]"))
            coefficients = self.block_transform(normalized_block)
            indicess[block_index], biggest_coefficients[block_index] = self.bin(coefficients)
        indicess = self.center(indicess)

        return CompressedTensor(tensor.shape, first_elements, biggest_coefficients, indicess)

    def decompress(self, compressed: CompressedTensor):
        decompressed = torch.empty(compressed.blocks_shape + self.block_shape, dtype=self.dtype, device=self.device)
        uncentered = self.center_inverse(compressed.indicess)

        for block_index in tqdm.tqdm(
            itertools.product(*(range(size) for size in compressed.blocks_shape)),
            desc="blockwise decompression",
            total=math.prod(compressed.blocks_shape),
        ):
            # index_range_str = ",".join(
            #     f"{block_index_element * block_size} : {block_index_element * block_size + block_size}"
            #     for block_index_element, block_size in zip(block_index, self.block_shape)
            # )
            start_time = time.time()
            coefficients = self.bin_inverse(uncentered[block_index], compressed.biggest_coefficients[block_index])
            self.bin_inverse_time += time.time() - start_time

            start_time = time.time()
            differences = self.block_transform(coefficients, inverse=True)
            self.block_transform_inverse_time += time.time() - start_time

            start_time = time.time()
            block = self.normalize_inverse(compressed.first_elements[block_index], differences)
            self.normalize_inverse_time += time.time() - start_time

            # exec(f"decompressed[{index_range_str}] = self.decompress_block(*compressed[block_index])")
            # exec(f"decompressed[{index_range_str}] = block")
            decompressed[block_index] = block

        decompressed = self.block_inverse(decompressed)
        return eval(f"decompressed[{','.join(f':{size}' for size in compressed.original_shape)}]").to(self.device)

    # def compress_block(self, block: torch.Tensor) -> tuple[float, torch.Tensor]:
    #     normalized_block = self.normalize(block)
    #     coefficients = self.block_transform(normalized_block)
    #     indices, biggest_coefficient = self.bin(coefficients)
    #     centered = self.center(indices)
    #     return biggest_coefficient, centered
    #
    # def decompress_block(self, first_element: float, biggest_coefficient: float, indices: torch.Tensor) -> torch.Tensor:
    #     uncentered = self.center_inverse(indices)
    #     coefficients = self.bin_inverse(uncentered, biggest_coefficient)
    #     differences = self.block_transform(coefficients, inverse=True)
    #     return self.normalize_inverse(first_element, differences)
    #
    # def dot_product_block(self, a: CompressedBlock, b: CompressedBlock, row: int, column: int) -> float:
    #     return self.decompress_block(a)[row] @ self.decompress_block(b)[:, column]

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
        for blocked_indices in tqdm.tqdm(
            itertools.product(*(range(size) for size in blocked.shape)), desc="unblocking", total=torch.numel(blocked)
        ):
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

    def normalize(self, block: torch.Tensor) -> torch.Tensor:
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
                                unnormalized,
                                torch.tensor(
                                    np.ravel_multi_index(adjacent_indices, self.block_shape), device=self.device
                                ),
                            )
                            / n_slice_indices
                        )

        return unnormalized

    def bin(self, coefficients: torch.Tensor) -> tuple[torch.Tensor, float]:
        """
        Section II.c

        :param coefficients:
        :return: Indices of the coefficient bins
        """
        biggest_coefficient = coefficients.norm(torch.inf)
        bins = torch.linspace(
            -biggest_coefficient - (2 * biggest_coefficient) / (self.n_bins - 2),
            biggest_coefficient,
            self.n_bins,
            dtype=self.dtype,
            device=self.device,
        )
        return (coefficients.unsqueeze(-1) - bins).abs().min(-1).indices.type(self.index_dtype), biggest_coefficient

    def bin_inverse(self, indices: torch.Tensor, biggest_coefficient: float) -> torch.Tensor:
        """
        Section II.(-c)

        :param indices: Indices of the bins
        :param biggest_coefficient: biggest bin value.
        :return: Values corresponding to each index
        """
        biggest_coefficient = float(biggest_coefficient)  # in case it's a Tensor.
        assert biggest_coefficient >= 0, f"The biggest coefficient {biggest_coefficient} should be non-negative."

        bins = torch.linspace(
            -biggest_coefficient - (2 * biggest_coefficient) / (self.n_bins - 2),
            biggest_coefficient,
            self.n_bins,
            dtype=self.dtype,
            device=self.device,
        )
        return bins[indices.type(torch.int64)]

    def center(self, slope_indices: torch.Tensor) -> torch.Tensor:
        """
        This adjustment follows binning and is not explicitly stated in the paper.
        We apply this adjustment trying to get closer results to the examples in the paper.

        :param slope_indices:
        :return: block of integers where the mean slope index is 0.
        """
        return slope_indices - self.n_bins // 2

    def center_inverse(self, centered_slope_indices: torch.Tensor) -> torch.Tensor:
        """
        Adjust slope indices to be non-negative.

        :param centered_slope_indices:
        :return: block of non-negative slope indices
        """
        return centered_slope_indices + self.n_bins // 2

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
            slope_indices.to(self.dtype),
            range(self.n_dimensions),
            self.transformer_tensor if not inverse else self.inverse_transformer_tensor,
            range(2 * self.n_dimensions),
            range(self.n_dimensions, 2 * self.n_dimensions),
        )

        return transformed

    def dot_product(
        self, compressed_a: CompressedTensor, compressed_b: CompressedTensor, row: int, column: int
    ) -> float:
        assert (
            compressed_a.n_dimensions == 2 and compressed_b.n_dimensions == 2
        ), "Dot product not defined for dimensions other than 2."

        a_block_index = row // self.block_shape[0]
        b_block_index = column // self.block_shape[1]
        block_row = row % self.block_shape[0]
        block_column = column % self.block_shape[1]

        return sum(
            self.dot_product_block(compressed_a[a_index], compressed_b[b_index], block_row, block_column)
            for a_index, b_index in zip(
                ((a_block_index, column_index) for column_index in range(compressed_a.blocks_shape[0])),
                ((row_index, b_block_index) for row_index in range(compressed_b.blocks_shape[1])),
            )
        )


if __name__ == "__main__":
    _test()
