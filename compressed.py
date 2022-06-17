import itertools

import torch
import numpy as np


def _test():
    from compression import Compressor
    dtype = torch.float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    compressor = Compressor(dtype=dtype, device=device)
    a = torch.tensor([[0.01 * x * y for y in range(8)] for x in range(8)], dtype=dtype, device=device)
    # b = torch.tensor([[0.02 * x * y for y in range(8)] for x in range(8)], dtype=dtype, device=device)

    compressed_a = compressor.compress(a)
    # compressed_b = compressor.compress(b)
    # decompressed_product = compressor.decompress(compressed_a * 2)
    #
    # print(a * 2)
    # print(decompressed_product)
    # print(((a * 2) - decompressed_product).norm(torch.inf))


class CompressedBlock:
    def __init__(self, indices: torch.Tensor, first_element: float, biggest_element: float):
        self.indices = indices
        self.first_element = first_element
        self.biggest_element = biggest_element

    def __neg__(self):
        """
        :return: negated compressed block
        """
        return CompressedBlock(self.indices, -self.first_element, self.biggest_element)

    def __add__(self, other):
        """
        :param other: compressed block
        :return: the compressed sum of self and other
        """
        a_indices = self.indices.type(torch.int64)
        b_indices = other.indices.type(torch.int64)

        indices = torch.zeros_like(self.indices, dtype=torch.int64)
        where_can_divide = a_indices + b_indices != 0
        indices[where_can_divide] = torch.div(
            a_indices[where_can_divide] * b_indices[where_can_divide],
            a_indices[where_can_divide] + b_indices[where_can_divide],
            rounding_mode="floor",
        )

        return CompressedBlock(
            indices.type(self.indices.dtype),
            self.first_element + other.first_element,
            self.biggest_element + other.biggest_element,
        )

    def __sub__(self, other):
        return self + -other

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return CompressedBlock(
                self.indices, self.first_element * other, self.biggest_element
            )
        elif isinstance(other, CompressedBlock):
            a_indices = self.indices.type(torch.int64)
            b_indices = other.indices.type(torch.int64)

            first_element = self.first_element * other.first_element
            biggest_element = self.biggest_element + other.biggest_element
            indices = torch.zeros_like(self.indices, dtype=torch.int64)
            where_can_divide = a_indices + b_indices != 0
            indices[where_can_divide] = torch.div(
                a_indices[where_can_divide] * b_indices[where_can_divide],
                a_indices[where_can_divide] + b_indices[where_can_divide],
                rounding_mode="floor",
            )
            return CompressedBlock(indices.type(self.indices.dtype), first_element, biggest_element)
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

    @property
    def transpose(self):
        """

        :return: blah
        """
        return self

    def __getitem__(self, item):
        return self.blocks[item]

    def __setitem__(self, key, value):
        self[key] = value

    def __neg__(self):
        blocks = np.ndarray(self.blocks_shape, dtype=object)
        for block_index in itertools.product(*(range(size) for size in self.blocks_shape)):
            blocks[block_index] = -self[block_index]
        return blocks

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
        if hasattr(other, "__getitem__"):
            for block_index in itertools.product(*(range(size) for size in self.blocks_shape)):
                blocks[block_index] = operation(self[block_index], other[block_index])
            return CompressedTensor(blocks)
        else:
            for block_index in itertools.product(*(range(size) for size in self.blocks_shape)):
                blocks[block_index] = operation(self[block_index], other)
            return CompressedTensor(blocks)


if __name__ == "__main__":
    _test()
