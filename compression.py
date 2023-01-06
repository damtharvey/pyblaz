import functools
import itertools
import math
import pathlib

import torch
import torch.nn.functional
import tqdm

import compressed
import transforms
from compressed import CompressedBlock, CompressedTensor


def _test():
    import argparse
    from tabulate import tabulate

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--dimensions", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=8, help="size of a hypercubic block")
    parser.add_argument("--max-size", type=int, default=256)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=(
            dtypes := {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
                "float64": torch.float64,
            }
        ),
    )
    parser.add_argument(
        "--index-dtype",
        type=str,
        default="int16",
        choices=(
            index_dtypes := {"int8": torch.int8, "int16": torch.int16, "int32": torch.int32, "int64": torch.int64}
        ),
    )
    args = parser.parse_args()

    dtype = dtypes[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    block_shape = (args.block_size,) * args.dimensions
    compressor = Compressor(
        block_shape=block_shape,
        dtype=dtype,
        index_dtype=index_dtypes[args.index_dtype],
        device=device,
    )

    table = []

    for size in tqdm.tqdm(
        tuple(1 << p for p in range(args.block_size.bit_length() - 1, args.max_size.bit_length())),
        desc=f"time {args.dimensions}D",
    ):
        results = [size]

        x = torch.randn((size,) * args.dimensions, dtype=dtype, device=device)
        y = torch.randn((size,) * args.dimensions, dtype=dtype, device=device)

        # compress
        compressed_x = compressor.compress(x)
        compressed_y = compressor.compress(y)
        results.append((compressor.decompress(compressed_x) - x).norm(torch.inf))

        # compressed negate
        results.append((compressor.decompress(-compressed_x) + x).norm(torch.inf))

        # compressed add
        results.append((compressor.decompress(compressed_x + compressed_y) - (x + y)).norm(torch.inf))

        # compressed multiply
        results.append((compressor.decompress(compressed_x * 3.14159) - (x * 3.14159)).norm(torch.inf))

        # compressed dot
        results.append(abs((compressed_x.dot(compressed_y) - (x * y).sum())))

        # compressed norm2
        results.append(abs(compressed_x.norm_2() - x.norm(2)))

        # compressed mean
        _ = compressed_x.mean()
        results.append(abs(compressed_x.mean() - x.mean()))

        # compressed variance
        results.append(abs(compressed_x.variance() - x.var(unbiased=False)))

        # compressed cosine similarity
        results.append(abs(compressed_x.cosine_similarity(compressed_y) - (x * y).sum() / (x.norm(2) * y.norm(2))))

        table.append(results)

    print(
        tabulate(
            table,
            headers=("size", "codec", "negate", "add", "multiply", "dot", "norm2", "mean", "variance", "cosine"),
        )
    )


class Compressor:
    """
    compressor inspired by https://arxiv.org/abs/2202.13007

    From communication with the author, instead of binning followed by orthogonal transform,
    later versions featured orthogonal transform followed by binning, which is what we implement here.

    We also make the normalize step optional and off by default.
    By skipping it, compressed space linear operations are facilitated.


    Other features include
    * Arbitrary dimensionality
    * GPU support
    * Variable data type for coefficient indices

    To do:
    * Dropping certain indices to save space. Currently, we use all indices.
    """

    def __init__(
        self,
        block_shape: tuple[int, ...] = (8, 8),
        transform: callable = transforms.cosine,
        dtype: torch.dtype = torch.float32,
        index_dtype: torch.dtype = torch.int8,
        mask: torch.BoolTensor = None,
        device: torch.device = torch.device("cuda"),
        transform_tensor_directory: pathlib.Path = pathlib.Path("temp") / "transform_tensors",
    ):
        self.block_shape = block_shape
        self.log_2_block_shape = tuple(size.bit_length() - 1 for size in self.block_shape)
        self.transform = transform
        self.n_dimensions = len(block_shape)
        self.dtype = dtype
        self.index_dtype = index_dtype
        self.mask = mask.to(device) if mask is not None else torch.ones(block_shape, dtype=torch.bool, device=device)
        self.device = device
        self.transform_tensor_directory = transform_tensor_directory

        self.n_coefficients = None
        self.transformer_tensor = None
        self.inverse_transformer_tensor = None

    def compress(self, tensor: torch.Tensor) -> CompressedTensor:
        """
        Compress the tensor.

        :param tensor: uncompressed tensor
        :returns: compressed tensor
        """
        assert self.n_dimensions == len(tensor.shape), (
            f"Compressor dimensionality ({self.n_dimensions}) "
            f"must match tensor dimensionality ({len(tensor.shape)})."
        )

        blocked = self.block(tensor)
        # If there is pruning, it is faster to calculate all coefficients and then drop some.
        indicess, biggest_coefficients = self.bin(self.blockwise_transform(blocked)[..., self.mask])
        return CompressedTensor(tensor.shape, biggest_coefficients, indicess, self.mask)

    def decompress(self, compressed_tensor: CompressedTensor):
        """
        This isn't blockwise complete decompression. Some things are better done over the whole tensor.
        """
        assert self.n_dimensions == compressed_tensor.n_dimensions, (
            f"Compressor dimensionality ({self.n_dimensions}) "
            f"must match tensor dimensionality ({compressed_tensor.n_dimensions})."
        )
        coefficientss = torch.zeros(
            compressed_tensor.blocks_shape + compressed_tensor.block_shape, dtype=self.dtype, device=self.device
        )
        coefficientss[..., compressed_tensor.mask] = self.bin_inverse(compressed_tensor)

        unblocked = self.block_inverse(self.blockwise_transform(coefficientss, inverse=True))

        return eval(f"unblocked[{','.join(f':{size}' for size in compressed_tensor.original_shape)}]")

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
        for intrablock_index in itertools.product(*(range(size) for size in self.block_shape)):
            selection_string = ",".join(
                f"{intrablock_index_element}::{block_size}"
                for intrablock_index_element, block_size in zip(intrablock_index, self.block_shape)
            )
            blocked[(...,) + intrablock_index] = eval(f"padded[{selection_string}]")

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
        for intrablock_index in itertools.product(*(range(size) for size in self.block_shape)):
            selection_string = ",".join(
                f"{intrablock_index_element}::{block_size}"
                for intrablock_index_element, block_size in zip(intrablock_index, self.block_shape)
            )
            exec(f"unblocked[{selection_string}] = blocked[(...,) + intrablock_index]")

        return unblocked

    def blockwise_transform(
        self, blocked_tensor: torch.Tensor, n_coefficients: int = None, inverse=False
    ) -> torch.Tensor:
        """
        Section II.d

        Transform the blocked tensor blockwise according to self.block_shape.

        :param blocked_tensor:
        :param n_coefficients:
        :param inverse:
        :return:
        """
        if not n_coefficients:
            n_coefficients = math.prod(self.block_shape)

        if n_coefficients == self.n_coefficients and (
            (self.transformer_tensor is not None and not inverse)
            or (self.inverse_transformer_tensor is not None and inverse)
        ):
            transformer_tensor = self.transformer_tensor if not inverse else self.inverse_transformer_tensor
        else:
            transform_tensor_path = self.transform_tensor_directory / (
                "x".join(str(size) for size in self.block_shape) + f"_{n_coefficients}c_{self.dtype.__repr__()[6:]}_"
                f"{'inverse_' if inverse else ''}{self.transform.__name__}_tensor.pth"
            )
            if transform_tensor_path.exists():
                transformer_tensor = torch.load(transform_tensor_path).to(self.device)
            elif n_coefficients < math.prod(self.block_shape):
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
            else:
                transform_matrices = {
                    block_size: torch.tensor(
                        [
                            [self.transform(block_size, element, frequency, inverse) for frequency in range(block_size)]
                            for element in range(block_size)
                        ],
                        dtype=self.dtype,
                        device=self.device,
                    )
                    for block_size in set(self.block_shape)
                }
                einsum_arguments = []
                for direction, block_size in enumerate(self.block_shape):
                    einsum_arguments.append(transform_matrices[block_size])
                    einsum_arguments.append((direction, direction + self.n_dimensions))

                transformer_tensor = torch.einsum(*einsum_arguments)

            if not inverse:
                self.transformer_tensor = transformer_tensor
            else:
                self.inverse_transformer_tensor = transformer_tensor
            self.n_coefficients = n_coefficients

            self.transform_tensor_directory.mkdir(parents=True, exist_ok=True)
            torch.save(transformer_tensor, transform_tensor_path)

        return torch.einsum(
            blocked_tensor,
            # blocks, intrablock
            tuple(range(2 * self.n_dimensions)),
            transformer_tensor,
            # intrablock, coefficients
            tuple(range(self.n_dimensions, 3 * self.n_dimensions)),
            # blocks, coefficients
            tuple(range(self.n_dimensions)) + tuple(range(2 * self.n_dimensions, 3 * self.n_dimensions)),
        )

    def bin(self, coefficientss: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Blockwise binning according to Compressor.block_shape and Compressor.index_dtype

        :returns: tuple of (bins, biggest coefficients).
        Bins are shaped (block indices, coefficient indices).
        Biggest coefficients are shaped (block indices).
        """
        biggest_coefficients = coefficientss.norm(torch.inf, -1)
        return (
            (coefficientss * (compressed.INDICES_RADIUS[self.index_dtype] / biggest_coefficients[..., None]))
            .round()
            .type(self.index_dtype)
        ), biggest_coefficients

    def bin_inverse(self, compressed_tensor: CompressedTensor) -> torch.Tensor:
        """
        Blockwise mapping bins to values.

        :returns: tuple of (bins, biggest coefficients).
        Bins are shaped (block indices, coefficient indices).
        Biggest coefficients are shaped (block indices).
        """
        return (
            compressed_tensor.indicess.type(self.dtype)
            * compressed_tensor.biggest_coefficients[..., None]
            / compressed.INDICES_RADIUS[compressed_tensor.indicess.dtype]
        )

    def compress_block(self, block: torch.Tensor) -> CompressedBlock:
        """
        Compress a single block.

        Not recommended if you can compress a whole tensor at once.

        :param block: tensor shaped as Compressor.block_shape
        :returns: compressed block
        """
        normalized_block = self.normalize_block(block)
        coefficients = self.blockwise_transform(normalized_block[(None,) * self.n_dimensions + (...,)]).view(
            self.block_shape
        )
        biggest_coefficient = coefficients.norm(torch.inf)
        indices = self.bin_block(coefficients, biggest_coefficient)
        return CompressedBlock(block[(0,) * self.n_dimensions], biggest_coefficient, indices)

    def decompress_block(self, block: CompressedBlock) -> torch.Tensor:
        """
        Decompress a single block.

        Not recommended if you can decompress a whole tensor at once.

        :param block: compressed block
        :returns: decompressed block
        """
        coefficients = self.bin_inverse_block(block.indices, block.biggest_coefficient)
        differences = self.blockwise_transform(coefficients[(None,) * self.n_dimensions + (...,)], inverse=True).view(
            self.block_shape
        )
        return self.normalize_inverse_block(block.first_element, differences)

    def normalize_block(self, block: torch.Tensor) -> torch.Tensor:
        """
        Section II.b

        Block Normalization

        :param block: a block of the input
        :return: Tuple of (the first element of the block, the mean slope, the normalized block)
        """
        differences = torch.zeros_like(block, dtype=self.dtype, device=self.device)
        for n_slice_indices in range(1, self.n_dimensions + 1):
            for slice_directions in self._slice_directions_combinations(n_slice_indices):
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

    def normalize_inverse_block(self, first_element: float, block: torch.Tensor) -> torch.Tensor:
        """
        Section inverse II.(-b)

        inverse of block normalization

        :param first_element: first element of the block before it was normalized
        :param block: a block of inverse predicted values
        :return: inverse of the normalized block
        """
        unnormalized = block.detach()
        unnormalized[(0,) * self.n_dimensions] = first_element
        for n_slice_indices in range(1, self.n_dimensions + 1):
            for slice_directions in self._slice_directions_combinations(n_slice_indices):
                index_groups = self._index_groups(slice_directions, self.block_shape)
                for index_group in index_groups:
                    assignee_indices = torch.tensor(index_group, dtype=torch.int64)
                    for direction in slice_directions:
                        adjacent_indices = assignee_indices.clone()
                        adjacent_indices[direction] -= 1
                        unnormalized[index_group] += unnormalized[tuple(adjacent_indices)] / n_slice_indices

        return unnormalized

    def bin_block(self, coefficients: torch.Tensor, biggest_coefficient: float) -> torch.Tensor:
        """
        Section II.c

        :param coefficients:
        :param biggest_coefficient:
        :return: Centered indices of the coefficient bins
        """
        return coefficients * compressed.INDICES_RADIUS[self.index_dtype] / biggest_coefficient

    def bin_inverse_block(self, indices: torch.Tensor, biggest_coefficient: float) -> torch.Tensor:
        """
        Section II.(-c)

        :param indices: Centered indices of the bins
        :param biggest_coefficient: biggest bin value.
        :return: Values corresponding to each index
        """
        assert biggest_coefficient >= 0, f"The biggest coefficient {biggest_coefficient} should be non-negative."

        return indices.type(self.dtype) * biggest_coefficient / compressed.INDICES_RADIUS[self.index_dtype]

    @functools.cache
    def _slice_directions_combinations(self, n_slice_indices):
        return tuple(itertools.combinations(range(self.n_dimensions), n_slice_indices))

    @functools.cache
    def _index_groups(self, slice_directions, shape):
        return tuple(
            tuple(group)[0]
            for _, group in itertools.groupby(
                itertools.product(
                    *(range(1, size) if direction in slice_directions else (0,) for direction, size in enumerate(shape))
                ),
                key=lambda x: sum(x),
            )
        )


if __name__ == "__main__":
    _test()
