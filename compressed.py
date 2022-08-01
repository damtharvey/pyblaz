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
    def __init__(self, first_element, biggest_coefficient, indices):
        self.first_element = first_element
        self.biggest_coefficient = biggest_coefficient
        self.indices = indices


class CompressedTensor:
    def __init__(
        self,
        original_shape: tuple[int, ...],
        first_elements: torch.Tensor,
        biggest_coefficients: torch.Tensor,
        indicess: torch.Tensor,
    ):
        self.original_shape = original_shape
        self.first_elements = first_elements
        self.biggest_coefficients = biggest_coefficients
        self.indicess = indicess

    @property
    def n_dimensions(self) -> int:
        return len(self.original_shape)

    @property
    def blocks_shape(self) -> tuple[int, ...]:
        return self.first_elements.shape

    @property
    def block_shape(self) -> tuple[int, ...]:
        return torch.tensor(self.original_shape) / torch.tensor(self.blocks_shape)

    def __getitem__(self, item: tuple[int, ...] or int) -> CompressedBlock:
        """
        :returns: compressed block at the indices
        """
        return CompressedBlock(self.first_elements[item], self.biggest_coefficients[item], self.indicess[item])

    def __setitem__(self, key: tuple[int, ...] or int, value: CompressedBlock):
        self.first_elements[key] = value.first_element
        self.biggest_coefficients[key] = value.biggest_coefficient
        self.indicess[key] = value.indices

    def __neg__(self):
        """
        :returns: negated compressed tensor.
        """
        return CompressedTensor(self.original_shape, -self.first_elements, self.biggest_coefficients, -self.indicess)

    def __add__(self, other):
        """
        :returns: sum of two compressed tensors
        """
        indices = (
            self.indicess * self.biggest_coefficients[(...,) + (None,) * self.n_dimensions]
            + other.indicess * other.biggest_coefficients[(...,) + (None,) * other.n_dimensions]
        )

        proportion_of_radius = (
            indices.norm(torch.inf, tuple(range(self.n_dimensions, 2 * self.n_dimensions)))
            / INDICES_RADIUS[self.indicess.dtype]
        )
        indices = torch.nan_to_num(
            (indices / proportion_of_radius[(...,) + (None,) * other.n_dimensions]).round().type(self.indicess.dtype)
        )
        return CompressedTensor(
            self.original_shape,
            self.first_elements + other.first_elements,
            proportion_of_radius,
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
                self.first_elements * other,
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


if __name__ == "__main__":
    _test()
