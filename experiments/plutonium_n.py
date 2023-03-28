from cProfile import label
from compression import Compressor
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import os
import torch

list = [665, 670, 675, 680, 686, 687, 689, 690, 692, 693, 694, 695, 699]
compressed_subtraction = []
subtraction = []
print("timestamp, difference, subtraction, (de)compressed subtraction")
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

    compressor = Compressor(block_shape=(16, 8), dtype=dtype, device=device)

    a = torch.FloatTensor(final_list0)
    b = torch.FloatTensor(final_list1)
    compressed_a = compressor.compress(a)
    compressed_b = compressor.compress(b)
    subtraction.append(abs(a - b))
    compressed_subtraction.append(abs(compressor.decompress(compressed_b - compressed_a)))
    # difference.append(abs(subtraction - compressed_subtraction))

    # print(str(list[l]), torch.mean(difference), torch.mean(subtraction), torch.mean(compressed_subtraction))
plt.xlabel("timestep")
plt.ylabel("L2 norm value")
plt.title("L2 norm difference for each timestep for n-density of Pu atom")
# plt.xticks(
#     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
#     [
#         "665",
#         "670",
#         "675",
#         "680",
#         "686",
#         "687",
#         "689",
#         "690",
#         "692",
#         "693",
#         "694",
#         "695",
#         "699",
#     ],
# )
plt.plot(list[:-1], subtraction)

# plt.xticks(
#     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
#     [
#         "665",
#         "670",
#         "675",
#         "680",
#         "686",
#         "687",
#         "689",
#         "690",
#         "692",
#         "693",
#         "694",
#         "695",
#         "699",
#     ],
# )

plt.plot(list[:-1], compressed_subtraction)
plt.legend()
plt.savefig("L2norm_n.png")
plt.show()
plt.close()
