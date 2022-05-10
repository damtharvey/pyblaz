import math


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
