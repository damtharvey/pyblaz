import numpy as np
from PIL import Image
import rawpy
from skimage import io, color

# For Adobe DNG image
list = [665, 670, 675, 680, 686, 687, 689, 690, 692, 693, 694, 695, 699]
for i in range(0, len(list)):

    # For raw image (without any headers)
    input_file = "./data/plutonium/n/n_density3D_00000" + str(list[i]) + ".raw"
    npimg = np.fromfile(input_file, dtype=np.uint8)

    imageSize = (40, 40, 66)

    npimg = npimg.reshape(imageSize)
    # npimg = np.squeeze(npimg, axis=2)
    array = npimg / 1023.0
    reshaped_array = array.reshape(1600, 66)
    print(reshaped_array)
    # np.savetxt("./data/plutonium/txt/n/" + str(list[i]) + ".csv", reshaped_array, delimiter=",")

    input_file = "./data/plutonium/p/p_density3D_00000" + str(list[i]) + ".raw"
    npimg = np.fromfile(input_file, dtype=np.uint8)

    imageSize = (40, 40, 66)

    npimg = npimg.reshape(imageSize)
    # npimg = np.squeeze(npimg, axis=2)
    array = npimg
    reshaped_array = array.reshape(1600, 66)
    print(sum(reshaped_array))
    # np.savetxt("./data/plutonium/txt/p/" + str(list[i]) + ".csv", reshaped_array, delimiter=",")
