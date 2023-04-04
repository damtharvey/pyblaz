from compression import Compressor
import matplotlib.pyplot as plt
import structural_similarity
import torch
import numpy as np
import matplotlib.colors as mcolors

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
cmap = plt.get_cmap("seismic")

pc00 = axs[0, 0].pcolormesh(
    eta,
    cmap=cmap,
    norm=mcolors.TwoSlopeNorm(vmin=min(eta.flatten()), vcenter=0, vmax=max(eta.flatten())),
)
axs[0, 0].set_title("height for float16 precision data")

pc01 = axs[0, 1].pcolormesh(
    eta2,
    cmap=cmap,
    norm=mcolors.TwoSlopeNorm(vmin=min(eta2.flatten()), vcenter=0, vmax=max(eta2.flatten())),
)

axs[0, 1].set_title("height for float32 precision data")

cmap = plt.get_cmap("seismic")
uncompress_diff = eta - eta2
pc10 = axs[1, 0].pcolormesh(
    uncompress_diff,
    cmap=cmap,
    norm=mcolors.TwoSlopeNorm(vmin=min(uncompress_diff.flatten()), vcenter=0, vmax=max(uncompress_diff.flatten())),
)
axs[1, 0].set_title("difference of float16 vs float32 for uncompressed heights")

dtype = torch.float32

device = torch.device("cpu")
compressor = Compressor(block_shape=(32, 32), dtype=dtype, device=device)

a = torch.FloatTensor(eta)
b = torch.FloatTensor(eta2)


compressed_a = compressor.compress(a)
compressed_b = compressor.compress(b)


diff = compressor.decompress(compressed_a - compressed_b)
# generate 2 2d grids for the x & y bounds
print(eta)

print(torch.mean(diff - eta - eta2))
pc11 = axs[1, 1].pcolormesh(
    diff,
    cmap=cmap,
    norm=mcolors.TwoSlopeNorm(vmin=min(diff.flatten()), vcenter=0, vmax=max(diff.flatten())),
)

axs[1, 1].set_title("difference of float16 vs float32 for compressed heights with block size (32, 32)")
plt.colorbar(pc00, ax=axs[0, 0])
plt.colorbar(pc01, ax=axs[0, 1])
plt.colorbar(pc10, ax=axs[1, 0])
plt.colorbar(pc11, ax=axs[1, 1])
plt.show()
