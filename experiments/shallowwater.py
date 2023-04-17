from pyblaz.compression import PyBlaz
import matplotlib.pyplot as plt
import numpy as np
import torch

timesteps = []

absolute_error_fastmathvsO3 = []
absolute_error_ftzvsO3 = []

absolute_error_fastmathvsO3_compressed = []
absolute_error_ftzvsO3_compressed = []

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

    txt_file1 = open("./data/ShallowWatersEquations/output/" + str(timestep) + ".txt", "r")
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

    txt_file2 = open("./data/ShallowWatersEquations/ftz/" + str(timestep) + ".txt", "r")
    file_content2 = txt_file2.read()

    content_list2 = file_content2.split("\n")

    x = 80
    list_of_lists2 = [content_list2[i : i + x] for i in range(0, len(content_list2), x)]
    newlist2 = []
    for word in list_of_lists2[0]:
        word = word.split(",")
        newlist2.append(word)

    newlist2 = newlist2[:-1]
    final_list2 = []
    for i in newlist2:
        temp = i[0].split(" ")
        temp2 = []
        for j in temp[:-1]:
            temp2.append(float(j))
        final_list2.append(temp2)

    dtype = torch.float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    compressor = PyBlaz(block_shape=(8, 8), dtype=dtype, device=device)

    a = torch.FloatTensor(final_list0)
    b = torch.FloatTensor(final_list1)
    c = torch.FloatTensor(final_list2)

    compressed_a = compressor.compress(a)
    compressed_b = compressor.compress(b)
    compressed_c = compressor.compress(c)

    subtraction_fastmathvsO3 = abs(b - a)
    subtraction_ftzvsO3 = abs(c - a)

    timesteps.append(timestep)

    absolute_error_fastmathvsO3.append(torch.mean(subtraction_fastmathvsO3))
    absolute_error_ftzvsO3.append(torch.mean(subtraction_ftzvsO3))

    decompressed_subtraction_fastmathvsO3 = abs(compressor.decompress(compressed_b - compressed_a))
    decompressed_subtraction_ftzvsO3 = abs(compressor.decompress(compressed_c - compressed_a))

    absolute_error_fastmathvsO3_compressed.append(torch.mean(decompressed_subtraction_fastmathvsO3))
    absolute_error_ftzvsO3_compressed.append(torch.mean(decompressed_subtraction_ftzvsO3))


plt.plot(
    np.asarray(timesteps),
    np.asarray(absolute_error_fastmathvsO3),
    label="fastmath vs O3 w/o compression",
    color="brown",
)
plt.plot(np.asarray(timesteps), np.asarray(absolute_error_ftzvsO3), label="ftz vs O3 w/o compression", color="black")

plt.plot(
    np.asarray(timesteps),
    np.asarray(absolute_error_fastmathvsO3_compressed),
    label="fastmath vs O3 compressed",
    color="cyan",
)
plt.plot(
    np.asarray(timesteps), np.asarray(absolute_error_ftzvsO3_compressed), label="ftz vs O3 compressed", color="green"
)

plt.title("Mean error of Shallow water equations")
plt.xlabel("Timesteps [0-20)")
plt.ylabel("Mean error value")
plt.legend()
plt.savefig("mean_error_graph_absolute_error.png")
plt.close()
