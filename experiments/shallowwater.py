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


coefficient_difference = []
weighted_coefficient_difference = []
absolute_error = []
relative_error = []
actual_error = []
actual_relative_error = []
timesteps = []
for timestep in range(1):
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

    coefficient_sum_blockwise = []
    for blocks_rowwise in range(coefficient_a.size()[0]):
        temp = []
        for blocks_columnwise in range(coefficient_a.size()[1]):
            res = []
            ordered_elements = diagonalOrder(np.array(coefficient_a[blocks_rowwise][blocks_columnwise]), 8, 8)
            for i in range(len(ordered_elements)):
                res.append(sum(ordered_elements[i]))
            temp.append(res)
        coefficient_sum_blockwise.append(temp)
    print(coefficient_sum_blockwise)
    print(len(coefficient_sum_blockwise))
    print(len(coefficient_sum_blockwise[0]))
    print(len(coefficient_sum_blockwise[0][0]))

    for blocks_rowwise in range(len(coefficient_sum_blockwise)):
        for blocks_columnwise in range(len(coefficient_sum_blockwise[blocks_rowwise])):
            print(coefficient_sum_blockwise[blocks_rowwise][blocks_columnwise][0])

    # print(weighted_coefficient_sum_a)

    """
    weighted_coefficient_sum_a += (
        0.8 * coefficient_a[blocks_rowwise][blocks_columnwise][row_in_block][0]
        + 0.1 * coefficient_a[blocks_rowwise][blocks_columnwise][row_in_block][1]
        + 0.05 * coefficient_a[blocks_rowwise][blocks_columnwise][row_in_block][2]
        + 0.02 * coefficient_a[blocks_rowwise][blocks_columnwise][row_in_block][3]
        + 0.01 * coefficient_a[blocks_rowwise][blocks_columnwise][row_in_block][4]
        + 0.013 * coefficient_a[blocks_rowwise][blocks_columnwise][row_in_block][5]
        + 0.005 * coefficient_a[blocks_rowwise][blocks_columnwise][row_in_block][6]
        + 0.002 * coefficient_a[blocks_rowwise][blocks_columnwise][row_in_block][7]
    )
    blocks_b = compressor.block(b)
    # differences_b = compressor.normalize(blocks_b)
    coefficient_b = compressor.blockwise_transform(blocks_b)
    weighted_coefficient_sum_b = 0.0
    for blocks_rowwise in range(coefficient_b.size()[0]):
        for blocks_columnwise in range(coefficient_b.size()[1]):
            for row_in_block in range(coefficient_b.size()[2]):
                weighted_coefficient_sum_b += (
                    0.8 * coefficient_b[blocks_rowwise][blocks_columnwise][row_in_block][0]
                    + 0.1 * coefficient_b[blocks_rowwise][blocks_columnwise][row_in_block][1]
                    + 0.05 * coefficient_b[blocks_rowwise][blocks_columnwise][row_in_block][2]
                    + 0.02 * coefficient_b[blocks_rowwise][blocks_columnwise][row_in_block][3]
                    + 0.013 * coefficient_b[blocks_rowwise][blocks_columnwise][row_in_block][4]
                    + 0.01 * coefficient_b[blocks_rowwise][blocks_columnwise][row_in_block][5]
                    + 0.005 * coefficient_b[blocks_rowwise][blocks_columnwise][row_in_block][6]
                    + 0.002 * coefficient_b[blocks_rowwise][blocks_columnwise][row_in_block][7]
                )

    compressed_a = compressor.compress(a)
    compressed_b = compressor.compress(b)

    subtraction = b - a
    decompressed_subtraction = compressor.decompress(compressed_b - compressed_a)
    timesteps.append(timestep)
    coefficient_difference.append((coefficient_b - coefficient_a).norm(float("inf")))
    weighted_coefficient_difference.append(abs(weighted_coefficient_sum_b - weighted_coefficient_sum_a))
    absolute_error.append(decompressed_subtraction.norm(float("inf")))
    relative_error_tensor = np.nan_to_num(decompressed_subtraction / compressor.decompress(compressed_a))
    relative_error.append(LA.norm(relative_error_tensor, np.inf))
    # np.max(np.where(a==0, a.min(), a)
    actual_error.append(subtraction.norm(float("inf")))
    actual_relative_error.append(LA.norm((np.nan_to_num(subtraction / a)), np.inf))

# print(relative_error)
# print(compressor.decompress(compressed_a))
plt.plot(np.asarray(timesteps), np.asarray(coefficient_difference), label="coefficient difference")
plt.plot(np.asarray(timesteps), np.asarray(weighted_coefficient_difference), label="weighted coefficient difference")
plt.title("Coefficient difference of Shallow water equations")
plt.xlabel("timestep")
plt.ylabel("L infinity error")
plt.legend()
plt.savefig("L_infinity_error_graph_coefficient_difference.png")
plt.close()

plt.plot(np.asarray(timesteps), np.asarray(absolute_error), label="absolute error")
plt.plot(np.asarray(timesteps), np.asarray(actual_error), label="actual error")
plt.title("absolute error vs actual error of Shallow water equations")
plt.xlabel("timestep")
plt.ylabel("L infinity error")
plt.legend()
plt.savefig("L_infinity_error_graph_absolute_error.png")
plt.close()

plt.plot(np.asarray(timesteps), np.asarray(relative_error), label="relative error")
plt.plot(np.asarray(timesteps), np.asarray(actual_relative_error), label="actual relative error")
plt.title("relative error vs actual relative error of Shallow water equations")
plt.xlabel("timestep")
plt.ylabel("L infinity error")
plt.legend()
plt.savefig("L_infinity_error_graph_relative_error.png")
plt.close()"""
