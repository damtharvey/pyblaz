import torch


INDICES_RADIUS = {
    torch.int8: (1 << 7) - 1,
    torch.int16: (1 << 15) - 1,
    torch.int32: (1 << 31) - 1,
    torch.int64: (1 << 63) - 1,
}


def _test():
    pass


class CompressedBlock:
    def __init__(self, biggest_coefficient, indices):
        self.biggest_coefficient = biggest_coefficient
        self.indices = indices


class CompressedTensor:
    def __init__(
        self,
        original_shape: tuple[int, ...],
        biggest_coefficients: torch.Tensor,
        indicess: torch.Tensor,
    ):
        self.original_shape = original_shape
        self.biggest_coefficients = biggest_coefficients
        self.indicess = indicess

    @property
    def n_dimensions(self) -> int:
        return len(self.original_shape)

    @property
    def blocks_shape(self) -> tuple[int, ...]:
        return self.biggest_coefficients.shape

    @property
    def block_shape(self) -> tuple[int, ...]:
        return self.indicess.shape[self.n_dimensions :]

    def __getitem__(self, item: tuple[int, ...] or int) -> CompressedBlock:
        """
        :returns: compressed block at the indices
        """
        return CompressedBlock(self.biggest_coefficients[item], self.indicess[item])

    def __setitem__(self, key: tuple[int, ...] or int, value: CompressedBlock):
        self.biggest_coefficients[key] = value.biggest_coefficient
        self.indicess[key] = value.indices

    def __neg__(self):
        """
        :returns: negated compressed tensor.
        """
        return CompressedTensor(self.original_shape, self.biggest_coefficients, -self.indicess)

    def __add__(self, other):
        """
        :returns: sum of two compressed tensors or of a compressed tensor and a scalar
        """
        if isinstance(other, CompressedTensor):
            assert self.original_shape == other.original_shape and self.block_shape == other.block_shape, (
                f"Original shapes and block shapes must match. "
                f"Got original shapes {self.original_shape} and {other.original_shape}, "
                f"block shapes {self.block_shape} and {other.block_shape}."
            )
            return self.add_tensor(other)
        elif isinstance(other, (float, int)) or (isinstance(other, torch.Tensor) and other.numel() == 1):
            return self.add_scalar(other)
        else:
            raise TypeError(f"Add not defined between {type(self)} and {type(other)}.")

    def add_tensor(self, other):
        indices = (
            self.indicess * self.biggest_coefficients[(...,) + (None,) * self.n_dimensions]
            + other.indicess * other.biggest_coefficients[(...,) + (None,) * other.n_dimensions]
        )
        proportion_of_radius = (
            indices.norm(torch.inf, tuple(range(self.n_dimensions, 2 * self.n_dimensions)))
            / INDICES_RADIUS[self.indicess.dtype]
        )
        indices = torch.nan_to_num(
            (indices / proportion_of_radius[(...,) + (None,) * self.n_dimensions]).round().type(self.indicess.dtype)
        )
        return CompressedTensor(
            self.original_shape,
            proportion_of_radius,
            indices,
        )

    def add_scalar(self, other):
        coefficientss = (
            self.indicess.type(self.biggest_coefficients.dtype)
            * self.biggest_coefficients[(...,) + (None,) * self.n_dimensions]
            / INDICES_RADIUS[self.indicess.dtype]
        )
        coefficientss[(...,) + (0,) * self.n_dimensions] += other * torch.prod(torch.tensor(self.block_shape) ** 0.5)
        biggest_coefficients = coefficientss.norm(torch.inf, tuple(range(self.n_dimensions, 2 * self.n_dimensions)))
        indices = (
            (
                coefficientss
                * (INDICES_RADIUS[self.indicess.dtype] / biggest_coefficients[(...,) + (None,) * self.n_dimensions])
            )
            .round()
            .type(self.indicess.dtype)
        )
        return CompressedTensor(
            self.original_shape,
            biggest_coefficients,
            indices,
        )

    def __sub__(self, other):
        """
        :returns: difference of two compressed tensors (self - other)
        """
        return self + -other

    def __mul__(self, other):
        """
        :returns: compressed tensor scaled by a scalar
        """
        if isinstance(other, (float, int)) or (isinstance(other, torch.Tensor) and other.numel() == 1):
            product = CompressedTensor(
                self.original_shape,
                self.biggest_coefficients * abs(other),
                self.indicess * (1 if other >= 0 else -1),
            )
            return product
        else:
            raise TypeError(f"Multiply not defined between {type(self)} and {type(other)}.")

    def __rmul__(self, other):
        """
        :returns: compressed tensor scaled by a scalar
        """
        return self * other

    def dot(self, other) -> float:
        """
        :returns: the dot product of this tensor with another compressed tensor.
        """
        self_biggest_coefficients_scaled = (
            self.biggest_coefficients[(...,) + (None,) * self.n_dimensions] / INDICES_RADIUS[self.indicess.dtype]
        )
        other_biggest_coefficients_scaled = (
            other.biggest_coefficients[(...,) + (None,) * other.n_dimensions] / INDICES_RADIUS[other.indicess.dtype]
        )
        return (
            (self.indicess.type(self.biggest_coefficients.dtype) * self_biggest_coefficients_scaled)
            * (other.indicess.type(other.biggest_coefficients.dtype) * other_biggest_coefficients_scaled)
        ).sum()

    def norm_2(self) -> float:
        """
        :returns: the L_2 norm.
        """
        # TODO: Compare with self.dot(self) ** 0.5

        self_biggest_coefficients_scaled = (
            self.biggest_coefficients[(...,) + (None,) * self.n_dimensions] / INDICES_RADIUS[self.indicess.dtype]
        )
        return (self.indicess.type(self.biggest_coefficients.dtype) * self_biggest_coefficients_scaled).norm(2)

    def cosine_similarity(self, other) -> float:
        return self.dot(other) / (self.norm_2() * other.norm_2())

    def mean(self) -> float:
        """
        :returns: the arithmetic mean of the compressed tensor.
        """
        biggest_coefficients_scaled = self.biggest_coefficients / INDICES_RADIUS[self.indicess.dtype]
        return (
            (
                self.indicess.type(self.biggest_coefficients.dtype)[(...,) + (0,) * self.n_dimensions]
                * biggest_coefficients_scaled
            ).sum()
            / torch.prod(torch.tensor(self.blocks_shape))
            / torch.prod(torch.tensor(self.block_shape) ** 0.5)
        )

    def variance(self, sample: bool = False) -> float:
        """
        :param sample: whether to return the sample variance
        :returns: the variance of the compressed tensor
        """
        biggest_coefficients_scaled = (
            self.biggest_coefficients[(...,) + (None,) * self.n_dimensions] / INDICES_RADIUS[self.indicess.dtype]
        )
        coefficientss = self.indicess.type(self.biggest_coefficients.dtype) * biggest_coefficients_scaled

        coefficientss[(...,) + (0,) * self.n_dimensions] -= coefficientss[
            (...,) + (0,) * self.n_dimensions
        ].sum() / torch.prod(torch.tensor(self.blocks_shape))

        variance = (coefficientss**2).mean()

        if sample:
            return variance * (n_elements := torch.prod(torch.tensor(self.original_shape))) / (n_elements - 1)
        else:
            return variance


if __name__ == "__main__":
    _test()
