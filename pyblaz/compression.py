import itertools
import pathlib

import torch
import torch.nn
import torch.nn.functional
import tqdm

import compressed
import transforms
from compressed import CompressedTensor


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
        choices=(index_dtypes := {"int8": torch.int8, "int16": torch.int16}),
    )
    args = parser.parse_args()

    dtype = dtypes[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    block_shape = (args.block_size,) * args.dimensions
    compressor = PyBlaz(
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

        # compressed add scalar
        results.append((compressor.decompress(compressed_x + 3.14159) - (x + 3.14159)).norm(torch.inf))

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

        # compressed covariance
        results.append(abs(((x - x.mean()) * (y - y.mean())).mean() - compressed_x.covariance(compressed_y)))

        table.append(results)

    print(
        tabulate(
            table,
            headers=(
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
            ),
        )
    )


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

        blocked = self.block(tensor)
        # If there is pruning, it is faster to calculate all coefficients and then drop some.
        indicess, biggest_coefficients = self.bin(self.codec.blockwise_transform(blocked)[..., self.codec.mask])
        return CompressedTensor(tensor.shape, biggest_coefficients, indicess, self.codec.mask)

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
                        for size, block_size in zip(reversed(unblocked.shape), reversed(self.codec.block_shape))
                    )
                )
            ),
        )

        blocks_shape = tuple(
            unblocked_size >> log_2_block_size
            for unblocked_size, log_2_block_size in zip(padded.shape, self.codec.log_2_block_shape)
        )

        blocked = torch.zeros(blocks_shape + self.codec.block_shape, dtype=self.codec.dtype, device=self.codec.device)
        for intrablock_index in itertools.product(*(range(size) for size in self.codec.block_shape)):
            selection_string = ",".join(
                f"{intrablock_index_element}::{block_size}"
                for intrablock_index_element, block_size in zip(intrablock_index, self.codec.block_shape)
            )
            blocked[(...,) + intrablock_index] = eval(f"padded[{selection_string}]")

        return blocked

    def bin(self, coefficientss: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Blockwise binning according to Compressor.block_shape and Compressor.index_dtype

        :returns: tuple of (bins, biggest coefficients).
        Bins are shaped (block indices, coefficient indices).
        Biggest coefficients are shaped (block indices).
        """
        biggest_coefficients = coefficientss.norm(torch.inf, -1)
        return (
            (coefficientss * (compressed.INDICES_RADIUS[self.codec.index_dtype] / biggest_coefficients[..., None]))
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
            compressed_tensor.blocks_shape + compressed_tensor.block_shape, dtype=self.codec.dtype, device=self.codec.device
        )
        coefficientss[..., compressed_tensor.mask] = self.bin_inverse(compressed_tensor)

        unblocked = self.block_inverse(self.codec.blockwise_transform(coefficientss, inverse=True))

        return eval(f"unblocked[{','.join(f':{size}' for size in compressed_tensor.original_shape)}]")

    def block_inverse(self, blocked: torch.Tensor) -> torch.Tensor:
        """
        Reshape the blocked form tensor into unblocked form.

        :param blocked: tensor of shape blocks' shape followed by block shape.
        :return: unblocked tensor
        """
        unblocked_shape = (
            *(n_blocks << size for n_blocks, size in zip(blocked.shape[: self.codec.n_dimensions], self.codec.log_2_block_shape)),
        )
        unblocked = torch.zeros(unblocked_shape, dtype=self.codec.dtype, device=self.codec.device)
        for intrablock_index in itertools.product(*(range(size) for size in self.codec.block_shape)):
            selection_string = ",".join(
                f"{intrablock_index_element}::{block_size}"
                for intrablock_index_element, block_size in zip(intrablock_index, self.codec.block_shape)
            )
            exec(f"unblocked[{selection_string}] = blocked[(...,) + intrablock_index]")

        return unblocked

    def bin_inverse(self, compressed_tensor: CompressedTensor) -> torch.Tensor:
        """
        Blockwise mapping bins to values.

        :returns: tuple of (bins, biggest coefficients).
        Bins are shaped (block indices, coefficient indices).
        Biggest coefficients are shaped (block indices).
        """
        return (
            compressed_tensor.indicess.type(self.codec.dtype)
            * compressed_tensor.biggest_coefficients[..., None]
            / compressed.INDICES_RADIUS[compressed_tensor.indicess.dtype]
        )


if __name__ == "__main__":
    _test()
