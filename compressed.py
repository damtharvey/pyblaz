import itertools

import torch
import numpy as np


def _test():
    pass


class CompressedBlock:
    INDICES_RADIUS = {
        torch.int8: (1 << 7) - 1,
        torch.int16: (1 << 15) - 1,
        torch.int32: (1 << 31) - 1,
        torch.int64: (1 << 63) - 1,
    }

    def __init__(self, first_element: float, biggest_coefficient: float, indices: torch.Tensor):
        self.first_element = first_element
        self.biggest_coefficient = biggest_coefficient
        self.indices = indices

    def __neg__(self):
        """
        :return: negated compressed block
        """
        return CompressedBlock(-self.first_element, self.biggest_coefficient, -self.indices)

    def __add__(self, other):
        """
        :param other: compressed block
        :return: the compressed sum of self and other
        """
        indices = self.indices * self.biggest_coefficient + other.indices * other.biggest_coefficient
        proportion_of_radius = indices.norm(torch.inf) / self.INDICES_RADIUS[self.indices.dtype]
        if proportion_of_radius:
            indices = (indices / proportion_of_radius).round().type(self.indices.dtype)
        return (
            self.first_element + other.first_element,
            proportion_of_radius,
            indices,
        )

    def __sub__(self, other):
        return self + -other

    def __mul__(self, other):
        if isinstance(other, (float, int)) or (isinstance(other, torch.Tensor) and other.numel() == 1):
            product = CompressedBlock(
                self.first_element * abs(other), self.biggest_coefficient * abs(other), self.indices
            )
            if other < 0:
                product = -product
            return product
        else:
            raise TypeError(f"Multiply not defined between {type(self)} and {type(other)}.")

    def __rmul__(self, other):
        return self * other


class CompressedTensor:
    INDICES_RADIUS = {
        torch.int8: (1 << 7) - 1,
        torch.int16: (1 << 15) - 1,
        torch.int32: (1 << 31) - 1,
        torch.int64: (1 << 63) - 1,
    }

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
        return torch.tensor(self.original_shape) / self.blocks_shape

    # def __getitem__(self, item):
    #     return CompressedBlock(self.first_elements[item], self.biggest_coefficients[item], self.indicess[item])

    def __getitem__(self, item):
        return self.first_elements[item], self.biggest_coefficients[item], self.indicess[item]

    def __setitem__(self, key, value):
        self[key] = value

    def __neg__(self):
        # blocks = np.ndarray(self.blocks_shape, dtype=object)
        # for block_index in itertools.product(*(range(size) for size in self.blocks_shape)):
        #     blocks[block_index] = -self[block_index]
        # return CompressedTensor(blocks, self.original_shape)
        return CompressedTensor(self.original_shape, -self.first_elements, self.biggest_coefficients, -self.indicess)

    def __add__(self, other):
        # return self.blockwise_binary(other, CompressedBlock.__add__)
        # indices = self.indicess * eval(
        #     f"self.biggest_coefficients[{':,' * self.n_dimensions + 'None,' * self.n_dimensions}]"
        # ) + other.indicess * eval(
        #     f"other.biggest_coefficients[{':,' * other.n_dimensions + 'None,' * other.n_dimensions}]"
        # )
        # proportion_of_radius = (
        #     indices.norm(torch.inf, tuple(range(self.n_dimensions, 2 * self.n_dimensions)))
        #     / self.INDICES_RADIUS[self.indicess.dtype]
        # )
        # indices = torch.nan_to_num(
        #     (indices / eval(f"proportion_of_radius[{':,' * other.n_dimensions + 'None,' * other.n_dimensions}]"))
        #     .round()
        #     .type(self.indicess.dtype),
        # )
        # return CompressedTensor(
        #     self.original_shape,
        #     self.first_elements + other.first_elements,
        #     proportion_of_radius,
        #     indices,
        # )

        biggest_coefficients = torch.zeros_like(self.biggest_coefficients)
        indicess = torch.zeros_like(self.indicess)

        for block_index in itertools.product(*(range(size) for size in self.blocks_shape)):
            indices = self.indicess[block_index] * self.biggest_coefficients[block_index] + other.indicess[block_index] * other.biggest_coefficients[block_index]
            proportion_of_radius = indices.norm(torch.inf) / self.INDICES_RADIUS[self.indicess.dtype]
            if proportion_of_radius:
                biggest_coefficients[block_index] = proportion_of_radius
                indicess[block_index] = (indices / proportion_of_radius).round().type(self.indicess.dtype)

        return CompressedTensor(
            self.original_shape,
            self.first_elements + other.first_elements,
            biggest_coefficients,
            indicess,
        )

    def __sub__(self, other):
        return self.blockwise_binary(other, CompressedBlock.__sub__)

    def __mul__(self, other):
        return self.blockwise_binary(other, CompressedBlock.__mul__)

    def __rmul__(self, other):
        return self * other

    def __matmul__(self, other):
        """
        :param other:
        :return: the matrix multiplication self @ other
        """
        pass

    def dot(self, other):
        """

        :param other:
        :return: the dot product between self and other
        """
        pass

    # def blockwise_binary(self, other, operation: callable):
    #     first_elements = torch.zeros_like(self.first_elements)
    #     biggest_coefficients = torch.zeros_like(self.biggest_coefficients)
    #     indicess = torch.zeros_like(self.indicess)
    #
    #     if isinstance(other, CompressedTensor):
    #         for block_index in itertools.product(*(range(size) for size in self.blocks_shape)):
    #             first_elements[block_index], biggest_coefficients[block_index], indicess[block_index] = operation(self[block_index], other[block_index])
    #         return result
    #     else:
    #         for block_index in itertools.product(*(range(size) for size in self.blocks_shape)):
    #             result[block_index] = operation(self[block_index], other)
    #         return result


if __name__ == "__main__":
    _test()
