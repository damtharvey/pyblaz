from compression import Compressor
import matplotlib.pyplot as plt

import torch
import numpy as np
import time


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def main():
    list = [665, 670, 675, 680, 686, 687, 689, 690, 692, 693, 694, 695, 699]
    print("timestamp, difference, subtraction, (de)compressed subtraction")
    list_compressed_wass = []
    list_wass = []
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
        device = torch.device("cpu")

        compressor = Compressor(block_shape=(8, 8), dtype=dtype, device=device)

        a = torch.FloatTensor(final_list0)
        b = torch.FloatTensor(final_list1)

        compressed_a = compressor.compress(a)
        compressed_b = compressor.compress(b)

        order = 53

        time_start = time.time()
        decompress_a = compressor.decompress(compressed_a)
        decompress_b = compressor.decompress(compressed_b)

        softmax_x = softmax(np.asarray(decompress_a))
        softmax_y = softmax(np.asarray(decompress_b))

        sorted_softmax_x = np.sort(softmax_x, axis=None)
        sorted_softmax_y = np.sort(softmax_y, axis=None)
        wass_distance = [
            ((abs(a - b)) ** order).mean() ** (1 / order) for a, b in zip(sorted_softmax_x, sorted_softmax_y)
        ]
        list_wass.append(np.mean(wass_distance))
        time_end = time.time()

        time_compress_start = time.time()
        compressed_x_mean = compressed_a.mean_blockwise()
        compressed_y_mean = compressed_b.mean_blockwise()

        softmax_compressed_x_mean = softmax(np.asarray(compressed_x_mean))
        softmax_compressed_y_mean = softmax(np.asarray(compressed_y_mean))

        sorted_softmax_compressed_x_mean = np.sort(softmax_compressed_x_mean, axis=None)
        sorted_softmax_compressed_y_mean = np.sort(softmax_compressed_y_mean, axis=None)

        wass_distance_compressed = [
            ((abs(a - b)) ** order).mean() ** (1 / order)
            for a, b in zip(sorted_softmax_compressed_x_mean, sorted_softmax_compressed_y_mean)
        ]

        list_compressed_wass.append(
            np.mean(wass_distance_compressed),
        )
        time_compress_end = time.time()
    print(list_compressed_wass)
    print(list_wass)
    print("time take with (de)compression=", time_end - time_start)
    print("time taken without (de)compression", time_compress_end - time_compress_start)
    print(
        "speedup",
        (abs(time_end - time_start) - abs(time_compress_end - time_compress_start)) * 100 / abs(time_end - time_start),
        "%",
    )
    print(max(list_compressed_wass), " ", list_compressed_wass.index(max(list_compressed_wass)))
    print(max(list_wass), " ", list_wass.index(max(list_wass)))
    plt.plot(list[:-1], list_compressed_wass, label="compressed")
    plt.plot(list[:-1], list_wass, label="(de)compressed")
    plt.show()


if __name__ == "__main__":
    main()
