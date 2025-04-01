import itertools
import pathlib

import torch
import torch.nn
import torch.nn.functional

from pyblaz.transforms import cosine
from pyblaz.compressed import CompressedTensor, INDICES_RADIUS


class PyBlaz:
    """
    Compressor inspired by https://arxiv.org/abs/2202.13007

    From communication with the author, instead of binning followed by orthogonal transform,
    later versions featured orthogonal transform followed by binning, which is what we implement here.

    We skip the normalize step. By skipping it, compressed space linear operations are facilitated.
    This allows direct operations on compressed tensors without needing to decompress first.

    Parameters
    ----------
    block_shape : tuple[int, ...]
        The shape of blocks to divide the tensor into for compression.
    transform : callable
        The transform function to use for compression. Default is cosine transform.
    dtype : torch.dtype
        The data type to use for the basis coefficients.
    index_dtype : torch.dtype
        The integer type to use for indices. Determines the precision of compression.
    mask : torch.Tensor, optional
        Boolean mask indicating which coefficients to keep. If None, all are kept.
    device : torch.device
        The device to perform computation on.
    transform_tensor_directory : pathlib.Path
        Directory to cache transform tensors for faster loading.
    compute_mode : str, optional
        The compute mode to use. Options are:
        - "fp32": Standard FP32 computation (default)
        - "tf32": Use TensorFloat32 for faster matrix operations on NVIDIA Ampere+ GPUs.
                 Requires dtype=torch.float32.
    compile : bool, optional
        Whether to compile the compression and decompression functions using torch.compile.
        Default is False.
    """

    def __init__(
        self,
        block_shape: tuple[int, ...] = (8, 8),
        transform: callable = cosine,
        dtype: torch.dtype = torch.float32,
        index_dtype: torch.dtype = torch.int8,
        mask: torch.Tensor = None,
        device: torch.device = torch.device("cuda"),
        transform_tensor_directory: pathlib.Path = pathlib.Path.home() / ".cache" / "pyblaz" / "transform_tensors",
        compute_mode: str = "fp32",
        compile: bool = False,
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

        # Configure compute mode settings
        if compute_mode == "tf32" and torch.cuda.is_available() and device.type == "cuda":
            if dtype != torch.float32:
                raise ValueError("TF32 compute mode can only be used with torch.float32 dtype")
            # Enable TF32 for matrix multiplications (available on Ampere+ GPUs)
            torch.backends.cuda.matmul.allow_tf32 = True
            # Enable TF32 for cuDNN operations
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, "autocast"):
                self.autocast_context = lambda: torch.autocast("cuda", dtype=dtype)
            else:
                # Fallback to a no-op context manager
                class NoOpContext:
                    def __enter__(self):
                        return None

                    def __exit__(self, *args):
                        return None

                self.autocast_context = lambda: NoOpContext()
        else:
            # If TF32 not requested or not supported, use a no-op context manager
            class NoOpContext:
                def __enter__(self):
                    return None

                def __exit__(self, *args):
                    return None

            self.autocast_context = lambda: NoOpContext()

        self.compressor = Compressor(self, *args, **kwargs)
        self.decompressor = Decompressor(self, *args, **kwargs)

        # Compile if requested
        if compile:
            if not torch.cuda.is_available():
                raise ValueError("Compilation requires CUDA to be available")
            self._compress = torch.compile(self.compress)
            self._decompress = torch.compile(self.decompress)
        else:
            self._compress = self.compress
            self._decompress = self.decompress

    def compress(self, tensor: torch.Tensor) -> CompressedTensor:
        return self.compressor(tensor)

    def decompress(self, compressed_tensor: CompressedTensor) -> torch.Tensor:
        return self.decompressor(compressed_tensor)

    def __call__(self, tensor: torch.Tensor) -> CompressedTensor:
        """Alias for compress to make the class callable."""
        return self._compress(tensor)

    def blockwise_transform(self, blocked_tensor: torch.Tensor, inverse=False) -> torch.Tensor:
        """
        Transform the blocked tensor blockwise according to self.block_shape.

        Parameters
        ----------
        blocked_tensor : torch.Tensor
            Tensor to transform in blocked form.
        inverse : bool, optional
            Whether to apply the inverse transform.

        Returns
        -------
        torch.Tensor
            Transformed tensor.
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

        # Use TF32 for the matrix multiplication if enabled
        with self.autocast_context():
            result = torch.einsum(
                blocked_tensor,
                # blocks, intrablock
                tuple(range(2 * self.n_dimensions)),
                transformer_tensor,
                # intrablock, coefficients
                tuple(range(self.n_dimensions, 3 * self.n_dimensions)),
                # blocks, coefficients
                tuple(range(self.n_dimensions)) + tuple(range(2 * self.n_dimensions, 3 * self.n_dimensions)),
            )

        return result


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

        Parameters
        ----------
        tensor : torch.Tensor
            Uncompressed tensor to compress.

        Returns
        -------
        CompressedTensor
            Compressed representation of the input tensor.
        """
        assert self.codec.n_dimensions == len(tensor.shape), (
            f"Compressor dimensionality ({self.codec.n_dimensions}) "
            f"must match tensor dimensionality ({len(tensor.shape)})."
        )

        blocked = self.block(tensor.to(self.codec.dtype).to(self.codec.device))

        # If there is pruning, it is faster to calculate all coefficients and then drop some.
        # Use TF32 acceleration for the blockwise transform if enabled
        coefficientss = self.codec.blockwise_transform(blocked)[..., self.codec.mask]

        del blocked
        indicess, biggest_coefficients = self.bin(coefficientss)
        del coefficientss
        return CompressedTensor(tensor.shape, biggest_coefficients, indicess, self.codec.mask)

    def block(self, unblocked: torch.Tensor) -> torch.Tensor:
        """
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
        Decompress a compressed tensor back to its original form.

        Parameters
        ----------
        compressed_tensor : CompressedTensor
            The compressed tensor to decompress.

        Returns
        -------
        torch.Tensor
            The decompressed tensor.
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

        # Use TF32 acceleration for the blockwise transform if enabled
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
