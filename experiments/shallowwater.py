import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

import torch
import tqdm

from compression import Compressor

timesteps = []

fastmath_vs_o3_norm2 = []
ftz_vs_o3_norm2 = []

fastmath_vs_o3_with_codec_norm2 = []
ftz_vs_o3_with_codec_norm2 = []

fastmath_vs_o3_compressed_norm2 = []
ftz_vs_o3_compressed_norm2 = []

dtype = torch.float64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

compressor = Compressor(block_shape=(8, 8), dtype=dtype, device=device)

for timestep in tqdm.tqdm(range(500)):
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

    a = torch.tensor(final_list0, dtype=dtype, device=device)
    b = torch.tensor(final_list1, dtype=dtype, device=device)
    c = torch.tensor(final_list2, dtype=dtype, device=device)

    compressed_a = compressor.compress(a)
    compressed_b = compressor.compress(b)
    compressed_c = compressor.compress(c)

    decompressed_a = compressor.decompress(compressed_a)
    decompressed_b = compressor.decompress(compressed_b)
    decompressed_c = compressor.decompress(compressed_c)

    timesteps.append(timestep)

    fastmath_vs_o3_norm2.append((b - a).norm(2).item())
    ftz_vs_o3_norm2.append((c - a).norm(2).item())

    # fastmath_vs_o3_with_codec_norm2.append((decompressed_b - decompressed_a).norm(2).item())
    # ftz_vs_o3_with_codec_norm2.append((decompressed_c - decompressed_a).norm(2).item())

    fastmath_vs_o3_compressed_norm2.append((compressed_b - compressed_a).norm_2().item())
    ftz_vs_o3_compressed_norm2.append((compressed_c - compressed_a).norm_2().item())

tab10 = list(matplotlib.colors.TABLEAU_COLORS.keys())


def draw_curves():
    # plt.plot(
    #     np.asarray(timesteps),
    #     np.asarray(fastmath_vs_o3_with_codec_norm2),
    #     label="fastmath vs O3 with (de)compression",
    #     # color=tab10[1],
    #     linewidth=1,
    # )
    # plt.plot(
    #     np.asarray(timesteps),
    #     np.asarray(ftz_vs_o3_with_codec_norm2),
    #     label="ftz vs O3 with (de)compression",
    #     # color=tab10[3],
    #     linewidth=1,
    # )

    plt.plot(np.asarray(timesteps), np.asarray(fastmath_vs_o3_norm2), label="fastmath vs O3",
             # color=tab10[0]
             )

    plt.plot(
        np.asarray(timesteps),
        np.asarray(fastmath_vs_o3_norm2),
        label="fastmath vs O3, compressed L2",
        # color=tab10[7],
        linewidth=1, linestyle=(0, (5, 5))
    )

    plt.plot(np.asarray(timesteps), np.asarray(ftz_vs_o3_norm2), label="ftz vs O3",
             # color=tab10[2]
             )

    plt.plot(
        np.asarray(timesteps),
        np.asarray(ftz_vs_o3_compressed_norm2),
        label="ftz vs O3, compressed L2",
        # color=tab10[5],
        linewidth=1, linestyle=(0, (5, 5))
    )


draw_curves()
plt.title("Magnitude of error in shallow water simulation")
plt.xlabel("time step")
plt.ylabel("magnitude of error")
plt.legend()
plt.savefig("results/ShallowWaters/ftz_fastmath_vs_o3_magnitude.pdf")


plt.clf()
draw_curves()
plt.title("(Zoomed) Magnitude of error in shallow water simulation")
plt.xlabel("time step")
plt.ylabel("magnitude of error")
plt.legend()
plt.xticks(range(20), [str(x) for x in range(20)])
plt.xlim((0, 19))

plt.savefig("results/ShallowWaters/ftz_fastmath_vs_o3_magnitude_xlim_0_19.pdf")
