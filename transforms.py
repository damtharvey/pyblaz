import math
import torch


def _test():
    his_dct_matrix = torch.tensor([0.3535534, 0.3535534, 0.3535534, 0.3535534, 0.3535534, 0.3535534, 0.3535534, 0.3535534,
                        0.490393, 0.415735, 0.277785, 0.0975452, -0.0975452, -0.277785, -0.415735, -0.490393,
                        0.46194, 0.191342, -0.191342, -0.46194, -0.46194, -0.191342, 0.191342, 0.46194,
                        0.415735, -0.0975452, -0.490393, -0.277785, 0.277785, 0.490393, 0.0975452, -0.415735,
                        0.353553, -0.353553, -0.353553, 0.353553, 0.353553, -0.353553, -0.353553, 0.353553,
                        0.277785, -0.490393, 0.0975452, 0.415735, -0.415735, -0.0975452, 0.490393, -0.277785,
                        0.191342, -0.46194, 0.46194, -0.191342, -0.191342, 0.46194, -0.46194, 0.191342,
                        0.0975452, -0.277785, 0.415735, -0.490393, 0.490393, -0.415735, 0.277785, -0.0975452]).reshape(8,8)
    my_dct_matrix = discrete_cosine_transform_matrix(8)
    
    print((his_dct_matrix - my_dct_matrix).norm(torch.inf))


def cosine(block_size: int, element: int, frequency: int, inverse: bool = False) -> float:
    """
    :param block_size: The number of elements in the vector you want to transform.
    :param element: Index of the basis vector component to get.
    :param frequency: Index of the basis vector to use.
    :param inverse: Whether to return the element from the inverse transform matrix.
    :return: The (element, frequency)th element of the discrete cosine transform matrix.
    """
    if inverse:
        element, frequency = frequency, element
    return math.sqrt((1 + (frequency > 0)) / block_size) * math.cos(
        (2 * element + 1) * frequency * math.pi / (2 * block_size)
    )


def discrete_cosine_transform_matrix(size: int, inverse: bool = False):
    return torch.tensor(
        [[cosine(size, element, frequency, inverse) for element in range(size)] for frequency in range(size)]
    )


if __name__ == "__main__":
    _test()
