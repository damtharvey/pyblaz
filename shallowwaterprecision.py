from compression import Compressor
import matplotlib.pyplot as plt

import torch
import numpy as np
import time


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


u = np.load("./data/ShallowWatersPrecisionData/Float16/u.npz")
v = np.load("./data/ShallowWatersPrecisionData/Float16/v.npz")
eta = np.load("./data/ShallowWatersPrecisionData/Float16/eta.npz")
sst = np.load("./data/ShallowWatersPrecisionData/Float16/sst.npz")

u2 = np.load("./data/ShallowWatersPrecisionData/Float32/u.npz")
v2 = np.load("./data/ShallowWatersPrecisionData/Float32/v.npz")
eta2 = np.load("./data/ShallowWatersPrecisionData/Float32/eta.npz")
sst2 = np.load("./data/ShallowWatersPrecisionData/Float32/sst.npz")

fig, axs = plt.subplots(2, 2)


axs[0, 0].pcolormesh(eta)
axs[0, 0].set_title("eta for float 16")

axs[0, 1].pcolormesh(eta2)
axs[0, 1].set_title("eta for float 32")


axs[1, 0].pcolormesh(abs(eta - eta2), cmap=plt.cm.Greys)
axs[1, 0].set_title("difference for uncompressed etas")

dtype = torch.float32

device = torch.device("cpu")
compressor = Compressor(block_shape=(2, 2), dtype=dtype, device=device)

a = torch.FloatTensor(eta)
b = torch.FloatTensor(eta2)

compressed_a = compressor.compress(a)
compressed_b = compressor.compress(b)

diff = abs(compressor.decompress(compressed_a - compressed_b))


print(torch.mean(abs(diff - abs(eta - eta2))))
axs[1, 1].pcolormesh(diff, cmap=plt.cm.Greys)
axs[1, 1].set_title("difference for compressed etas")
plt.show()
