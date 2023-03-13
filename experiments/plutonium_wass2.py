from cProfile import label
from compression import Compressor
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import os
import torch
import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def pearson_coeff(x, y):
    sorted_x = np.sort(x)
    sorted_y = np.sort(y)
    mean_x = x.mean()
    mean_y = y.mean()
    numerator = [(a - mean_x) * (b - mean_y) for a, b in zip(sorted_x, sorted_y)]
    denominator1 = [(a - mean_x) ** 2 for a in sorted_x]
    denominator2 = [(b - mean_y) ** 2 for b in sorted_y]
    denominator = (sum(denominator1) * sum(denominator2)) ** (1 / 2)
    return sum(numerator) / denominator


list = [665, 670, 675, 680, 686, 687, 689, 690, 692, 693, 694, 695, 699]
print("timestamp, difference, subtraction, (de)compressed subtraction")
list_result = []
for l in range(0, len(list) - 1):

    txt_file0 = open("./data/plutonium/txt/n/" + str(list[l]) + ".csv", "r")
    file_content0 = txt_file0.read()

    content_list0 = file_content0.split("\n")
    x = 66
    list_of_lists0 = [content_list0[i : i + x] for i in range(0, len(content_list0), x)]
    newlist0 = []
    for word in list_of_lists0[0]:
        word = word.split(",")
        newlist0.append(word)

    newlist0 = newlist0[:-1]

    final_list0 = []
    for i in newlist0:
        temp1 = []
        for j in i:
            temp1.append(float(j))
        final_list0.append(temp1)
    # print(final_list0)

    txt_file1 = open("./data/plutonium/txt/n/" + str(list[l + 1]) + ".csv", "r")
    file_content1 = txt_file1.read()

    content_list1 = file_content1.split("\n")
    x = 66
    list_of_lists1 = [content_list1[i : i + x] for i in range(0, len(content_list1), x)]
    newlist1 = []
    for word in list_of_lists1[0]:
        word = word.split(",")
        newlist1.append(word)

    newlist1 = newlist1[:-1]

    final_list1 = []
    for i in newlist1:
        temp1 = []
        for j in i:
            temp1.append(float(j))
        final_list1.append(temp1)

    dtype = torch.float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    compressor = Compressor(block_shape=(32, 32), dtype=dtype, device=device)

    a = torch.FloatTensor(final_list0)
    b = torch.FloatTensor(final_list1)
    print(a.shape)
    compressed_a = compressor.compress(a)
    compressed_b = compressor.compress(b)

    compressed_x_mean = compressed_a.mean_blockwise()
    compressed_y_mean = compressed_b.mean_blockwise()
    print(compressed_x_mean.shape)
    compressed_x_variance = compressed_a.variance_blockwise()
    compressed_y_variance = compressed_b.variance_blockwise()

    softmax_compressed_x_mean = softmax(np.asarray(compressed_x_mean))
    softmax_compressed_y_mean = softmax(np.asarray(compressed_y_mean))

    softmax_compressed_x_variance = softmax(np.asarray(compressed_x_variance))
    softmax_compressed_y_variance = softmax(np.asarray(compressed_y_variance))

    sorted_softmax_compressed_x_mean = np.sort(softmax_compressed_x_mean)
    sorted_softmax_compressed_y_mean = np.sort(softmax_compressed_y_mean)

    sorted_softmax_compressed_x_variance = np.sort(softmax_compressed_x_variance)
    sorted_softmax_compressed_y_variance = np.sort(softmax_compressed_y_variance)

    sorted_softmax_compressed_x_mean = np.sort(softmax_compressed_x_mean)
    sorted_softmax_compressed_y_mean = np.sort(softmax_compressed_y_mean)

    sorted_softmax_compressed_x_variance = np.sort(softmax_compressed_x_variance)
    sorted_softmax_compressed_y_variance = np.sort(softmax_compressed_y_variance)

    wass_distance_mean = [
        (a - b) ** 2 for a, b in zip(sorted_softmax_compressed_x_mean, sorted_softmax_compressed_y_mean)
    ]

    wass_distance_varaince = [
        (a - b) ** 2 for a, b in zip(sorted_softmax_compressed_x_variance, sorted_softmax_compressed_y_variance)
    ]

    p_value = pearson_coeff(softmax_compressed_x_mean, softmax_compressed_y_mean)

    wass_distance_pearson = (
        2 * sorted_softmax_compressed_x_variance * sorted_softmax_compressed_y_variance * (1 - p_value)
    )

    wass_distance_compressed = (
        np.mean(wass_distance_mean) + np.mean(wass_distance_varaince) + np.mean(wass_distance_pearson)
    )

    list_result.append(
        wass_distance_compressed,
    )
print(list_result)
print(max(list_result))
plt.plot(list[:-1], list_result, label="2-Wasserstein")
plt.show()
