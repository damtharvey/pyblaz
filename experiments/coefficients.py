import argparse

from sqlalchemy import false

import compression
import torch
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", type=int, default=2)
    parser.add_argument(
        "--block-size", type=int, default=8, help="size of a hypercubic block"
    )
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
    coefficients_x = compressor.compress(x).coefficientss
    print(coefficients_x)


if __name__ == "__main__":
    main()
