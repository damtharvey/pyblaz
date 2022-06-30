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
        return CompressedBlock(
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
    def __init__(self, blocks: np.array):
        self.blocks = blocks

    @property
    def n_dimensions(self) -> int:
        return len(self.blocks.shape)

    @property
    def blocks_shape(self) -> tuple[int, ...]:
        return self.blocks.shape

    @property
    def block_shape(self) -> tuple[int, ...]:
        return self.blocks[(0,) * self.n_dimensions].shape

    def __getitem__(self, item):
        return self.blocks[item]

    def __setitem__(self, key, value):
        self[key] = value

    def __neg__(self):
        blocks = np.ndarray(self.blocks_shape, dtype=object)
        for block_index in itertools.product(*(range(size) for size in self.blocks_shape)):
            blocks[block_index] = -self[block_index]
        return CompressedTensor(blocks)

    def __add__(self, other):
        return self.blockwise_binary(other, CompressedBlock.__add__)

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
        return self.transpose @ other

    def blockwise_binary(self, other, operation: callable):
        blocks = np.ndarray(self.blocks_shape, dtype=object)
        if isinstance(other, CompressedTensor):
            for block_index in itertools.product(*(range(size) for size in self.blocks_shape)):
                blocks[block_index] = operation(self[block_index], other[block_index])
            return CompressedTensor(blocks)
        else:
            for block_index in itertools.product(*(range(size) for size in self.blocks_shape)):
                blocks[block_index] = operation(self[block_index], other)
            return CompressedTensor(blocks)


if __name__ == "__main__":
    _test()
