import argparse



import compression
import torch
from datetime import datetime
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=8, help="size of a hypercubic block")
    parser.add_argument(
        "--index-dtype",
        type=str,
        default="int8",
        choices=(
            index_dtypes := {
                "int8": torch.int8,
                "int16": torch.int16,
                "int32": torch.int32,
                "int64": torch.int64,
            }
        ),
    )
    args = parser.parse_args()

    dtype = torch.float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    compressor = compression.Compressor(
        block_shape=(args.block_size,) * args.dimensions,
        dtype=dtype,
        index_dtype=index_dtypes[args.index_dtype],
        device=device,
    )

    x = torch.rand(80, 80, dtype=dtype, device=device)
    blocks_x = compressor.block(x)
 

    # differences_x = compressor.normalize(blocks_x)
    coefficient_x = compressor.blockwise_transform(blocks_x)
    sum1 = 0.0
    for k in range(10):
        for i in range(8):
            for j in range(8):
                sum1 += 0.8*coefficient_x[0][k][i][0] + 0.1*coefficient_x[0][k][i][1] + 0.05*coefficient_x[0][k][i][2] + 0.02*coefficient_x[0][k][i][3] + 0.01*coefficient_x[0][k][i][4] + 0.013*coefficient_x[0][k][i][5] + 0.005*coefficient_x[0][k][i][6] + 0.002*coefficient_x[0][k][i][7]
    y = torch.rand(80, 80, dtype=dtype, device=device)
    blocks_y = compressor.block(y)
    # differences_y = compressor.normalize(blocks_y)
    coefficient_y = compressor.blockwise_transform(blocks_y)
    sum2 = 0.0
    for k in range(10):
        for i in range(8):
            for j in range(8):
                sum2 += 0.8*coefficient_y[0][k][i][0] + 0.1*coefficient_y[0][k][i][1] + 0.05*coefficient_y[0][k][i][2] + 0.02*coefficient_y[0][k][i][3] + 0.01*coefficient_y[0][k][i][4] + 0.013*coefficient_y[0][k][i][5] + 0.005*coefficient_y[0][k][i][6] + 0.002*coefficient_y[0][k][i][7]


    print(sum2 - sum1)


if __name__ == "__main__":
    main()
