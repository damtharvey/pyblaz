from cProfile import label
from compression import Compressor
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA

import torch


absolute_error = []
relative_error = []
actual_error = []
actual_relative_error = []
timesteps = []

for timestep in range(500):
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

    compressed_a = compressor.compress(a)
    compressed_b = compressor.compress(b)

    subtraction = b - a

    timesteps.append(timestep)
    decompressed_subtraction = compressor.decompress(compressed_b - compressed_a)
    absolute_error.append(decompressed_subtraction.norm(float("inf")))
    relative_error_tensor = np.nan_to_num(decompressed_subtraction / compressor.decompress(compressed_a))
    relative_error.append(LA.norm(relative_error_tensor, np.inf))
    # np.max(np.where(a==0, a.min(), a)
    actual_error.append(subtraction.norm(float("inf")))
    actual_relative_error.append(LA.norm((np.nan_to_num(subtraction / a)), np.inf))


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
plt.close()
