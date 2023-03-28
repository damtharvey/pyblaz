from compression import Compressor
import matplotlib.pyplot as plt
import structural_similarity
import torch
import numpy as np
import time

from matplotlib.pyplot import figure


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
cmap = plt.get_cmap("PiYG")

pc00 = axs[0, 0].pcolormesh(eta, cmap=cmap)
axs[0, 0].set_title("height for float16 precision data")

pc01 = axs[0, 1].pcolormesh(eta2, cmap=cmap)

axs[0, 1].set_title("height for float32 precision data")

cmap = plt.get_cmap("PiYG")
pc10 = axs[1, 0].pcolormesh(abs(eta - eta2), cmap=cmap)
axs[1, 0].set_title("difference of float16 vs float32 for compressed heights")

dtype = torch.float32

device = torch.device("cpu")
compressor = Compressor(block_shape=(2, 2), dtype=dtype, device=device)

a = torch.FloatTensor(eta)
b = torch.FloatTensor(eta2)


compressed_a = compressor.compress(a)
compressed_b = compressor.compress(b)


diff = abs(compressor.decompress(compressed_a - compressed_b))
# generate 2 2d grids for the x & y bounds
print(eta)

print(torch.mean(abs(diff - abs(eta - eta2))))
pc11 = axs[1, 1].pcolormesh(diff, cmap=cmap)

axs[1, 1].set_title("difference of float16 vs float32 for compressed heights")
plt.colorbar(pc00, ax=axs[0, 0])
plt.colorbar(pc01, ax=axs[0, 1])
plt.colorbar(pc10, ax=axs[1, 0])
plt.colorbar(pc11, ax=axs[1, 1])
plt.show()
