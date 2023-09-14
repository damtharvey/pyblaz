import math


def _test():
    pass


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


def haar(block_size: int, point: float, order: int, inverse: bool = False) -> float:
    """
    :param block_size: The number of elements in the vector you want to transform.
    :param point: Point / block size of the Haar function to sample.
    :param order: Order of the Haar function.
    :param inverse: Whether to return the element from the inverse transform matrix.
    :return: Haar function evaluated at the point / block size.
    """
    if inverse:
        point, order = order, point
    if not order:
        return 1 / math.sqrt(block_size)
    else:
        point /= block_size
        p = int(math.floor(math.log2(order)))
        q = order % (1 << p)
        if q / 2**p <= point < (q + 0.5) / 2**p:
            return 2 ** (p / 2) / math.sqrt(block_size)
        elif (q + 0.5) / 2**p <= point < (q + 1) / 2**p:
            return -(2 ** (p / 2)) / math.sqrt(block_size)
        else:
            return 0


if __name__ == "__main__":
    _test()
