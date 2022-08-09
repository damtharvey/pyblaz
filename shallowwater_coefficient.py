from cProfile import label
from compression import Compressor
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA

import torch


def diagonalOrder(arr, n, m):

    ordering_elements = [[] for i in range(n + m - 1)]

    for i in range(m):
        for j in range(n):
            ordering_elements[i + j].append(arr[j][i])

    return ordering_elements


weighted_sum_coefficients = []

first_coeff_diff = []
coefficient_difference = []
first_coefficient_difference = []
timesteps = []
for timestep in range(10):
    txt_file0 = open("./data/ShallowWatersEquations/output/" + str(timestep) + ".txt", "r")
    file_content0 = txt_file0.read()

    content_list0 = file_content0.split("\n")

    x = 80
    list_of_lists0 = [content_list0[i : i + x] for i in range(0, len(content_list0), x)]
    newlist0 = []
    for word in list_of_lists0[0]:
        word = word.split(",")
        newlist0.append(word)

    newlist0 = newlist0[:-1]
    final_list0 = []
    for i in newlist0:
        temp = i[0].split(" ")
        temp1 = []
        for j in temp[:-1]:
            temp1.append(float(j))
        final_list0.append(temp1)

    txt_file1 = open("./data/ShallowWatersEquations/fastmath_output/" + str(timestep) + ".txt", "r")
    file_content1 = txt_file1.read()

    content_list1 = file_content1.split("\n")

    x = 80
    list_of_lists1 = [content_list1[i : i + x] for i in range(0, len(content_list1), x)]
    newlist1 = []
    for word in list_of_lists1[0]:
        word = word.split(",")
        newlist1.append(word)

    newlist1 = newlist1[:-1]
    final_list1 = []
    for i in newlist1:
        temp = i[0].split(" ")
        temp1 = []
        for j in temp[:-1]:
            temp1.append(float(j))
        final_list1.append(temp1)

    dtype = torch.float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    compressor = Compressor(block_shape=(8, 8), dtype=dtype, device=device)

    a = torch.FloatTensor(final_list0)
    b = torch.FloatTensor(final_list1)

    blocks_a = compressor.block(a)
    # differences_a = compressor.normalize(blocks_a)
    coefficient_a = compressor.blockwise_transform(blocks_a)
    first_coefficient_a = coefficient_a[..., 0, 0]

    coefficient_sum_blockwise_a = []
    for blocks_rowwise in range(coefficient_a.size()[0]):
        temp = []
        for blocks_columnwise in range(coefficient_a.size()[1]):
            res = []
            ordered_elements = diagonalOrder(np.array(coefficient_a[blocks_rowwise][blocks_columnwise]), 8, 8)
            for i in range(len(ordered_elements)):
                res.append(sum(ordered_elements[i]))
            temp.append(res)
        coefficient_sum_blockwise_a.append(temp)

    weighted_sum_coefficients_a = np.zeros(15)

    weight_a = 1.0
    for blocks_rowwise in range(len(coefficient_sum_blockwise_a)):
        for blocks_columnwise in range(len(coefficient_sum_blockwise_a[blocks_rowwise])):
            for every_element in range(len(coefficient_sum_blockwise_a[blocks_rowwise][blocks_columnwise])):
                weighted_sum_coefficients_a[every_element] += (
                    weight_a * coefficient_sum_blockwise_a[blocks_rowwise][blocks_columnwise][every_element]
                )
                weight_a /= 2
    print(sum(weighted_sum_coefficients_a))

    blocks_b = compressor.block(b)
    # differences_b = compressor.normalize(blocks_b)
    coefficient_b = compressor.blockwise_transform(blocks_b)
    first_coefficient_b = coefficient_b[..., 0, 0]

    coefficient_sum_blockwise_b = []
    for blocks_rowwise in range(coefficient_b.size()[0]):
        temp = []
        for blocks_columnwise in range(coefficient_b.size()[1]):
            res = []
            ordered_elements = diagonalOrder(np.array(coefficient_b[blocks_rowwise][blocks_columnwise]), 8, 8)
            for i in range(len(ordered_elements)):
                res.append(sum(ordered_elements[i]))
            temp.append(res)
        coefficient_sum_blockwise_b.append(temp)

    weighted_sum_coefficients_b = np.zeros(15)
    weight_b = 1.0
    for blocks_rowwise in range(len(coefficient_sum_blockwise_b)):
        for blocks_columnwise in range(len(coefficient_sum_blockwise_b[blocks_rowwise])):
            for every_element in range(len(coefficient_sum_blockwise_b[blocks_rowwise][blocks_columnwise])):
                weighted_sum_coefficients_b[every_element] += (
                    weight_b * coefficient_sum_blockwise_b[blocks_rowwise][blocks_columnwise][every_element]
                )
                weight_b /= 2

    timesteps.append(timestep)
    coefficient_difference.append((coefficient_b - coefficient_a).norm(float("inf")))
    first_coefficient_difference.append((first_coefficient_b - first_coefficient_a).norm(float("inf")))
    first_coeff_diff.append(first_coefficient_b - first_coefficient_a)
    weighted_sum_coefficients.append((weighted_sum_coefficients_b - weighted_sum_coefficients_a).max())
    # print(np.unravel_index(np.argmax(first_coeff_diff, axis=None), first_coeff_diff.shape), first_coeff_diff.max())


# print(relative_error)
# print(compressor.decompress(compressed_a))
plt.plot(np.asarray(timesteps), np.asarray(coefficient_difference), label="coefficient difference")
plt.plot(np.asarray(timesteps), np.asarray(first_coefficient_difference), label="first coefficient difference")
plt.plot(np.asarray(timesteps), np.asarray(weighted_sum_coefficients), label="weighted coefficient difference")
plt.title("Coefficient difference of Shallow water equations")
plt.xlabel("timestep")
plt.ylabel("L infinity error")
plt.legend()
plt.savefig("L_infinity_error_graph_coefficient_difference.png")
plt.close()
