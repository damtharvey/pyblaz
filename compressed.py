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
        mask: torch.Tensor = None,
    ):
        self.original_shape = original_shape
        self.biggest_coefficients = biggest_coefficients
        self.indicess = indicess
        self.mask = mask

    @property
    def n_dimensions(self) -> int:
        return len(self.original_shape)

    @property
    def blocks_shape(self) -> tuple[int, ...]:
        return self.biggest_coefficients.shape

    @property
    def block_shape(self) -> tuple[int, ...]:
        return self.mask.shape

    @property
    def coefficientss(self) -> torch.Tensor:
        """
        :returns: coefficient blocks of the compressed tensor, obtained through partial decompression
        """
        coefficients_blocks = (
            self.indicess.type(self.biggest_coefficients.dtype)
            * self.biggest_coefficients[..., None]
            / INDICES_RADIUS[self.indicess.dtype]
        )
        if coefficients_blocks.isnan().any():
            coefficients_blocks = (
                self.biggest_coefficients[..., None]
                / INDICES_RADIUS[self.indicess.dtype]
                * self.indicess.type(self.biggest_coefficients.dtype)
            )
        return coefficients_blocks

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
        return CompressedTensor(self.original_shape, self.biggest_coefficients, -self.indicess, self.mask)

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
        assert torch.equal(self.mask, other.mask), "Masks must match between tensors that will be added."

        indices = (
            self.indicess * self.biggest_coefficients[..., None]
            + other.indicess * other.biggest_coefficients[..., None]
        )
        proportion_of_radius = indices.norm(torch.inf, -1) / INDICES_RADIUS[self.indicess.dtype]
        indices = torch.nan_to_num((indices / proportion_of_radius[..., None]).round().type(self.indicess.dtype))
        return CompressedTensor(self.original_shape, proportion_of_radius, indices, self.mask)

    def add_scalar(self, other):
        coefficientss = (
            self.indicess.type(self.biggest_coefficients.dtype)
            * self.biggest_coefficients[..., None]
            / INDICES_RADIUS[self.indicess.dtype]
        )
        coefficientss[..., 0] += other * torch.prod(torch.tensor(self.block_shape) ** 0.5)
        biggest_coefficients = coefficientss.norm(torch.inf, -1)
        indices = (
            (coefficientss * (INDICES_RADIUS[self.indicess.dtype] / biggest_coefficients[..., None]))
            .round()
            .type(self.indicess.dtype)
        )
        return CompressedTensor(self.original_shape, biggest_coefficients, indices, self.mask)

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
                self.indicess if other >= 0 else -self.indicess,
                self.mask,
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
        assert torch.equal(self.mask, other.mask), "Masks must match between tensors that will be dotted...for now."

        dot_product = (
            (
                self.indicess.type(self.biggest_coefficients.dtype)
                * (self.biggest_coefficients[..., None] / INDICES_RADIUS[self.indicess.dtype])
            )
            * (
                other.indicess.type(other.biggest_coefficients.dtype)
                * (other.biggest_coefficients[..., None] / INDICES_RADIUS[other.indicess.dtype])
            )
        ).sum()
        if not dot_product.isnan():
            return dot_product
        else:
            return (
                (
                    (self.biggest_coefficients[..., None] / INDICES_RADIUS[self.indicess.dtype])
                    * self.indicess.type(self.biggest_coefficients.dtype)
                )
                * (
                    (other.biggest_coefficients[..., None] / INDICES_RADIUS[other.indicess.dtype])
                    * other.indicess.type(other.biggest_coefficients.dtype)
                )
            ).sum()

    def log(self):
        # log_indicess = torch.log(self.indicess)
        log_coeff = torch.log(self.biggest_coefficients)
        return CompressedTensor(self.original_shape, log_coeff, self.indicess, self.mask)

    def norm_2(self) -> float:
        """
        :returns: the L_2 norm.
        """
        # Faster than self.dot(self) ** 0.5
        magnitude = (
            self.indicess.type(self.biggest_coefficients.dtype)
            * self.biggest_coefficients[..., None]
            / INDICES_RADIUS[self.indicess.dtype]
        ).norm(2)
        if not magnitude.isnan():
            return magnitude
        else:
            return (
                self.biggest_coefficients[..., None]
                / INDICES_RADIUS[self.indicess.dtype]
                * self.indicess.type(self.biggest_coefficients.dtype)
            ).norm(2)

    def cosine_similarity(self, other) -> float:
        return self.dot(other) / (self.norm_2() * other.norm_2())

    def mean(self) -> float:
        """
        :returns: the arithmetic mean of the compressed tensor.
        """
        first_coefficients_sum = (
            self.indicess.type(self.biggest_coefficients.dtype)[..., 0]
            * self.biggest_coefficients
            / INDICES_RADIUS[self.indicess.dtype]
        ).sum()

        if not first_coefficients_sum.isnan():
            return (
                first_coefficients_sum
                / torch.prod(torch.tensor(self.blocks_shape))
                / torch.prod(torch.tensor(self.block_shape) ** 0.5)
            )
        else:
            return (
                (
                    self.biggest_coefficients
                    / INDICES_RADIUS[self.indicess.dtype]
                    * self.indicess.type(self.biggest_coefficients.dtype)[..., 0]
                ).sum()
                / torch.prod(torch.tensor(self.blocks_shape))
                / torch.prod(torch.tensor(self.block_shape) ** 0.5)
            )

    def mean_blockwise(self) -> torch.tensor:
        """
        :returns: the blockwise mean of the compressed tensor.
        """

        first_coefficients_sum = (
            self.indicess.type(self.biggest_coefficients.dtype)[..., 0]
            * self.biggest_coefficients
            / INDICES_RADIUS[self.indicess.dtype]
        )

        if not torch.isnan(first_coefficients_sum).any():
            return first_coefficients_sum / torch.prod(torch.tensor(self.block_shape) ** 0.5)

        else:
            return (
                self.biggest_coefficients
                / INDICES_RADIUS[self.indicess.dtype]
                * self.indicess.type(self.biggest_coefficients.dtype)[..., 0]
            ) / torch.prod(torch.tensor(self.block_shape) ** 0.5)

    def covariance(self, other, sample: bool = False) -> float:
        """
        :param other: other CompressedTensor to return covariance with
        :param sample: whether to return the sample covariance
        :returns: the covariance of this CompressedTensor with another
        """
        assert torch.equal(self.mask, other.mask), "Masks must match between tensors to get covariance...for now."

        self_coefficientss = self.coefficientss
        other_coefficientss = other.coefficientss

        self_coefficientss[..., 0] -= self_coefficientss[..., 0].sum() / torch.prod(torch.tensor(self.blocks_shape))
        other_coefficientss[..., 0] -= other_coefficientss[..., 0].sum() / torch.prod(torch.tensor(other.blocks_shape))

        covariance = (self_coefficientss * other_coefficientss).mean()
        if sample:
            return covariance * (n_elements := torch.prod(torch.tensor(self.original_shape))) / (n_elements - 1)
        else:
            return covariance

    def variance(self, sample: bool = False) -> float:
        """
        :param sample: whether to return the sample variance
        :returns: the variance of the compressed tensor
        """
        # Faster than self.covariance(self)
        coefficientss = self.coefficientss
        coefficientss[..., 0] -= coefficientss[..., 0].sum() / torch.prod(torch.tensor(self.blocks_shape))

        variance = (coefficientss**2).mean()
        if sample:
            return variance * (n_elements := torch.prod(torch.tensor(self.original_shape))) / (n_elements - 1)
        else:
            return variance

    def standard_deviation(self, sample: bool = False) -> float:
        return self.variance(sample) ** 0.5

    def structural_similarity(
        self,
        other,
        luminance_weight: float = 1,
        contrast_weight: float = 1,
        structure_weight: float = 1,
        dynamic_range: float = 0,
        luminance_stabilization: float = 0.01,
        contrast_stabilization: float = 0.03,
    ) -> float:
        """
        Return the structural similarity index between compressed tensors.

        This was originally intended for measuring visual similarity between images.
        This is an extension to floating point arrays.

        :param other: compressed tensor to compare with
        :param luminance_weight: weight on mean measures
        :param contrast_weight: weight on variance measures
        :param structure_weight: weight on covariance measures
        :param dynamic_range: originally the maximum pixel value.
                              Probably most useful if you know the range of the uncompressed arrays beforehand.
        :param luminance_stabilization: luminance stabilization hyperparameter
        :param contrast_stabilization: contrast stabilization hyperparameter
        :returns: structural similarity index between compressed tensors.
        """
        self_mean = self.mean()
        other_mean = other.mean()
        self_variance = self.variance()
        other_variance = other.variance()
        self_standard_deviation = self_variance**0.5
        other_standard_deviation = other_variance**0.5
        covariance = self.covariance(other)

        luminance_stabilizer = luminance_stabilization * dynamic_range
        contrast_stabilizer = contrast_stabilization * dynamic_range
        similarity_stabilizer = contrast_stabilizer / 2

        luminance_similarity = (2 * self_mean * other_mean + luminance_stabilizer) / (
            self_mean**2 + other_mean**2 + luminance_stabilizer
        )
        contrast_similarity = (2 * self_standard_deviation * other_standard_deviation + contrast_stabilizer) / (
            self_variance + other_variance + contrast_stabilizer
        )
        structure_similarity = (covariance + similarity_stabilizer) / (
            self_standard_deviation * other_standard_deviation + similarity_stabilizer
        )

        return (
            luminance_similarity**luminance_weight
            + contrast_similarity**contrast_weight
            + structure_similarity**structure_weight
        )

    def variance_blockwise(self, sample: bool = False) -> torch.Tensor:
        """
        :param sample: whether to return the sample variance
        :returns: the blockwise variance matrix of the  compressed tensor
        """
        coefficientss = self.coefficientss
        variance = (coefficientss**2).mean(-1)

        if sample:
            return variance * (n_elements := torch.prod(torch.tensor(self.block_shape))) / (n_elements - 1)
        else:
            return variance

    def standard_deviation_blockwise(self, sample: bool = False) -> torch.Tensor:
        return self.variance_blockwise(sample) ** 0.5


if __name__ == "__main__":
    _test()
