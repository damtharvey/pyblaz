import itertools
import pathlib

import torch
import torch.nn
import torch.nn.functional

import transforms
from compressed import CompressedTensor, INDICES_RADIUS


def _test():
    import argparse
    import time
    import tqdm
    from tabulate import tabulate

    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=8, help="size of a hypercubic block")
    parser.add_argument("--max-size", type=int, default=512)
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
        choices=(index_dtypes := {"int8": torch.int8, "int16": torch.int16}),
    )
    args = parser.parse_args()

    dtype = dtypes[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    block_shape = (args.block_size,) * args.dimensions

    # n_coefficients = int(math.prod(block_shape) * 0.5)
    # mask = torch.zeros(block_shape, dtype=torch.bool)
    # for index in sorted(
    #     itertools.product(*(range(size) for size in block_shape)),
    #     key=lambda coordinates: sum(coordinates),
    # )[:n_coefficients]:
    #     mask[index] = True

    compressor = PyBlaz(
        block_shape=block_shape,
        dtype=dtype,
        index_dtype=index_dtypes[args.index_dtype],
        # mask=mask,
        device=device,
    )

    time_table = []
    error_table = []
    headers = (
        "size",
        "codec",
        "negate",
        "add",
        "add_scalar",
        "multiply",
        "dot",
        "norm2",
        "mean",
        "variance",
        "cosine",
        "covariance",
    )

    for size in tqdm.tqdm(
        tuple(1 << p for p in range(args.block_size.bit_length() - 1, args.max_size.bit_length())),
        desc=f"{args.dimensions}D",
    ):
        size += 1
        time_results = [size]
        error_results = [size]

        x = torch.randn((size,) * args.dimensions, dtype=dtype, device=device) - 1
        y = torch.randn((size,) * args.dimensions, dtype=dtype, device=device) + 2

        # compress
        start_time = time.time()
        compressed_x = compressor.compress(x)
        compressed_y = compressor.compress(y)
        time_results.append(time.time() - start_time)

        # compressed negate
        start_time = time.time()
        r = -compressed_x
        time_results.append(time.time() - start_time)
        error_results.append((compressor.decompress(r) + x).norm(torch.inf))

        # compressed add
        start_time = time.time()
        r = compressed_x + compressed_y
        time_results.append(time.time() - start_time)
        error_results.append((compressor.decompress(r) - (x + y)).norm(torch.inf))

        # compressed add scalar
        start_time = time.time()
        r = compressed_x + 3.14159
        time_results.append(time.time() - start_time)
        error_results.append((compressor.decompress(r) - (x + 3.14159)).norm(torch.inf))

        # compressed multiply
        start_time = time.time()
        r = compressed_x * 3.14159
        time_results.append(time.time() - start_time)
        error_results.append((compressor.decompress(r) - (x * 3.14159)).norm(torch.inf))

        # compressed dot
        start_time = time.time()
        r = compressed_x.dot(compressed_y)
        time_results.append(time.time() - start_time)
        error_results.append(abs((r - (x * y).sum())))

        # compressed norm2
        start_time = time.time()
        r = compressed_x.norm_2()
        time_results.append(time.time() - start_time)
        error_results.append(abs(r - x.norm(2)))

        # compressed mean
        start_time = time.time()
        r = compressed_x.mean()
        time_results.append(time.time() - start_time)
        error_results.append(abs(r - x.mean()))

        # compressed variance
        start_time = time.time()
        r = compressed_x.variance()
        time_results.append(time.time() - start_time)
        error_results.append(abs(r - x.var(unbiased=False)))

        # compressed cosine similarity
        start_time = time.time()
        r = compressed_x.cosine_similarity(compressed_y)
        time_results.append(time.time() - start_time)
        error_results.append(abs(r - (x * y).sum() / (x.norm(2) * y.norm(2))))

        # compressed covariance
        start_time = time.time()
        r = compressed_x.covariance(compressed_y)
        time_results.append(time.time() - start_time)
        error_results.append(abs(r - ((x - x.mean()) * (y - y.mean())).mean()))

        time_table.append(time_results)
        error_table.append(error_results)

    print("\ntime\n" + tabulate(time_table, headers=headers))

    print("\nerror\n" + tabulate(error_table, headers=headers))


class PyBlaz:
    """
    compressor inspired by https://arxiv.org/abs/2202.13007

    From communication with the author, instead of binning followed by orthogonal transform,
    later versions featured orthogonal transform followed by binning, which is what we implement here.

    We also make the normalize step optional and off by default.
    By skipping it, compressed space linear operations are facilitated.
    """

    def __init__(
        self,
        block_shape: tuple[int, ...] = (8, 8),
        transform: callable = transforms.cosine,
        dtype: torch.dtype = torch.float32,
        index_dtype: torch.dtype = torch.int8,
        mask: torch.Tensor = None,
        device: torch.device = torch.device("cuda"),
        transform_tensor_directory: pathlib.Path = pathlib.Path("temp") / "transform_tensors",
        *args,
        **kwargs,
    ):
        self.block_shape = block_shape
        self.log_2_block_shape = tuple(size.bit_length() - 1 for size in block_shape)
        self.transform = transform
        self.n_dimensions = len(block_shape)
        self.dtype = dtype
        self.index_dtype = index_dtype
        self.device = device
        self.mask = (
            mask.type(torch.bool).to(device)
            if mask is not None
            else torch.ones(block_shape, dtype=torch.bool, device=device)
        )

        self.transform_tensor_directory = transform_tensor_directory
        self.transformer_tensor = None
        self.inverse_transformer_tensor = None

        self.compressor = Compressor(self, *args, **kwargs)
        self.decompressor = Decompressor(self, *args, **kwargs)

    def compress(self, tensor: torch.Tensor) -> CompressedTensor:
        return self.compressor(tensor)

    def decompress(self, compressed_tensor: CompressedTensor) -> torch.Tensor:
        return self.decompressor(compressed_tensor)

    def blockwise_transform(self, blocked_tensor: torch.Tensor, inverse=False) -> torch.Tensor:
        """
        Section II.d

        Transform the blocked tensor blockwise according to self.block_shape.

        :param blocked_tensor:
        :param inverse:
        :return:
        """
        if (self.transformer_tensor is not None and not inverse) or (
            self.inverse_transformer_tensor is not None and inverse
        ):
            transformer_tensor = self.transformer_tensor if not inverse else self.inverse_transformer_tensor
        else:
            transform_tensor_path = self.transform_tensor_directory / (
                "x".join(str(size) for size in self.block_shape) + f"_{self.dtype.__repr__()[6:]}_"
                f"{'inverse_' if inverse else ''}{self.transform.__name__}_tensor.pth"
            )
            if transform_tensor_path.exists():
                transformer_tensor = torch.load(transform_tensor_path).to(self.device)
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


class Compressor(torch.nn.Module):
    def __init__(
        self,
        codec: PyBlaz,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.codec = codec

    def forward(self, tensor):
        """
        Compress the tensor.

        :param tensor: uncompressed tensor
        :returns: compressed tensor
        """
        assert self.codec.n_dimensions == len(tensor.shape), (
            f"Compressor dimensionality ({self.codec.n_dimensions}) "
            f"must match tensor dimensionality ({len(tensor.shape)})."
        )

        blocked = self.block(tensor.to(self.codec.dtype).to(self.codec.device))
        # If there is pruning, it is faster to calculate all coefficients and then drop some.
        coefficientss = self.codec.blockwise_transform(blocked)[..., self.codec.mask]
        del blocked
        indicess, biggest_coefficients = self.bin(coefficientss)
        del coefficientss
        return CompressedTensor(tensor.shape, biggest_coefficients, indicess, self.codec.mask)

    def block(self, unblocked: torch.Tensor) -> torch.Tensor:
        """
        Section II.a
        Block Splitting
        :param unblocked: uncompressed tensor
        :return: tensor of shape blocks' shape followed by block shape.
        """
        reversed_block_shape = tuple(reversed(self.codec.block_shape))
        stack = torch.nn.functional.pad(
            unblocked,
            tuple(
                itertools.chain(
                    *(
                        (0, (block_size - size) % block_size)
                        for size, block_size in zip(reversed(unblocked.shape), reversed_block_shape)
                    )
                )
            ),
        )
        for block_size in reversed_block_shape:
            stack = torch.stack(torch.split(stack, block_size, self.codec.n_dimensions - 1))
        return stack

    def bin(self, coefficientss: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Blockwise binning according to Compressor.block_shape and Compressor.index_dtype

        :returns: tuple of (bins, biggest coefficients).
        Bins are shaped (block indices, coefficient indices).
        Biggest coefficients are shaped (block indices).
        """
        biggest_coefficients = coefficientss.norm(torch.inf, -1)
        return (
            (coefficientss * (INDICES_RADIUS[self.codec.index_dtype] / biggest_coefficients[..., None]))
            .round()
            .type(self.codec.index_dtype)
        ), biggest_coefficients


class Decompressor(torch.nn.Module):
    def __init__(
        self,
        codec: PyBlaz,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.codec = codec

    def forward(self, compressed_tensor):
        """
        This isn't blockwise complete decompression. Some things are better done over the whole tensor.
        """
        assert self.codec.n_dimensions == compressed_tensor.n_dimensions, (
            f"Decompressor dimensionality ({self.codec.n_dimensions}) "
            f"must match tensor dimensionality ({compressed_tensor.n_dimensions})."
        )
        coefficientss = torch.zeros(
            compressed_tensor.blocks_shape + compressed_tensor.block_shape,
            dtype=self.codec.dtype,
            device=self.codec.device,
        )
        coefficientss[..., compressed_tensor.mask] = compressed_tensor.specified_coefficientss()

        unblocked = self.block_inverse(self.codec.blockwise_transform(coefficientss, inverse=True))
        del coefficientss
        if unblocked.shape != compressed_tensor.original_shape:
            for dimension in range(self.codec.n_dimensions):
                unblocked = unblocked.split(compressed_tensor.original_shape[dimension], dimension)[0]
        return unblocked

    def block_inverse(self, blocked: torch.Tensor) -> torch.Tensor:
        """
        Reshape the blocked form tensor into unblocked form.

        :param blocked: tensor of shape blocks' shape followed by block shape.
        :return: unblocked tensor
        """
        unblocked = blocked
        for dimension in range(self.codec.n_dimensions):
            unblocked = torch.cat(
                tuple(unblocked[block_index] for block_index in range(blocked.shape[dimension])),
                -self.codec.n_dimensions + dimension,
            )
        return unblocked


if __name__ == "__main__":
    _test()
