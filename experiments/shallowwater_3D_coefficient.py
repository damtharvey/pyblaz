from cProfile import label
from compression import Compressor
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA

import torch
from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns

import random
from scipy.stats import gaussian_kde

coefficients = []

timesteps = []
time = [0, 100, 200, 300, 400, 499]
for timestep in range(len(time)):
    txt_file0 = open("./data/ShallowWatersEquations/output/" + str(time[timestep]) + ".txt", "r")
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

    txt_file1 = open("./data/ShallowWatersEquations/fastmath_output/" + str(time[timestep]) + ".txt", "r")
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

    compressor = Compressor(block_shape=(128, 128), dtype=dtype, device=device)

    a = torch.FloatTensor(final_list0)
    b = torch.FloatTensor(final_list1)

    blocks_a = compressor.block(a)
    # differences_a = compressor.normalize(blocks_a)
    coefficient_a = compressor.blockwise_transform(blocks_a)

    blocks_b = compressor.block(b)
    # differences_b = compressor.normalize(blocks_b)
    coefficient_b = compressor.blockwise_transform(blocks_b)

    timesteps.append(time[timestep])
    print(timestep)
    coefficients.append(coefficient_b[0][0])

fig = plt.figure()


ax = fig.add_subplot(projection="3d")

ax.set_xlabel("Coefficient number")


ax.set_ylabel("Coefficient Value")


ax.set_zlabel("Timestep")


for plot in range(len(coefficients)):
    colors = ["#" + "".join([random.choice("ABCDEF0123456789") for i in range(6)])]
    # plt.hist(, density=True)
    # plt.show()
    # density = gaussian_kde(sum(coefficients[plot].numpy().flatten()))

    ax.plot3D(range(0, 16384), coefficients[plot].numpy().flatten(), timesteps[plot], c=colors[0])

plt.show()
