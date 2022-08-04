import argparse

from sqlalchemy import false

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

    y = torch.rand(80, 80, dtype=dtype, device=device)
    blocks_y = compressor.block(y)
    # differences_y = compressor.normalize(blocks_y)
    coefficient_y = compressor.blockwise_transform(blocks_y)

    print((coefficient_y - coefficient_x).max())


if __name__ == "__main__":
    main()
