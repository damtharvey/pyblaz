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

    print(
        "size,compress,add,compressed_add,subtract,compressed_subtract,multiply,compressed_multiply,decompress"
    )

    # warmup
    warmup(compressor, device, dtype)

    for size in (1 << p for p in range(3, 15)):
        x = torch.rand(size, size, dtype=dtype, device=device)
        y = torch.rand(size, size, dtype=dtype, device=device)

        # compress
        start_time = datetime.now()
        compressed_x = compressor.compress(x)
        compressed_y = compressor.compress(y)
        compress_time = ((datetime.now() - start_time) / 2).microseconds

        # add
        start_time = datetime.now()
        result = x + y
        add_time = (datetime.now() - start_time).microseconds

        # compressed add
        start_time = datetime.now()
        result = compressed_x + compressed_y
        compressed_add_time = (datetime.now() - start_time).microseconds

        # subtract
        start_time = datetime.now()
        result = x - y
        subtract_time = (datetime.now() - start_time).microseconds

        # compressed subtract
        start_time = datetime.now()
        compressed_sum = compressed_x - compressed_y
        compressed_subtract_time = (datetime.now() - start_time).microseconds

        # multiply
        start_time = datetime.now()
        result = x * 3.14159
        multiply_time = (datetime.now() - start_time).microseconds

        # compressed multiply
        start_time = datetime.now()
        result = compressed_x * torch.pi
        compressed_multiply_time = (datetime.now() - start_time).microseconds

        start_time = datetime.now()
        decompressed = compressor.decompress(compressed_x)
        decompress_time = (datetime.now() - start_time).microseconds

        print(
            f"{size},{compress_time},{add_time},{compressed_add_time},{subtract_time},{compressed_subtract_time},"
            f"{multiply_time},{compressed_multiply_time},{decompress_time}"
        )


def warmup(compressor, device, dtype):
    x = torch.rand(8, 8, dtype=dtype, device=device)
    y = torch.rand(8, 8, dtype=dtype, device=device)
    # compress
    start_time = datetime.now()
    compressed_x = compressor.compress(x)
    compressed_y = compressor.compress(y)
    compress_time = ((datetime.now() - start_time) / 2).microseconds
    # add
    start_time = datetime.now()
    result = x + y
    add_time = (datetime.now() - start_time).microseconds
    # compressed add
    start_time = datetime.now()
    result = compressed_x + compressed_y
    compressed_add_time = (datetime.now() - start_time).microseconds
    # subtract
    start_time = datetime.now()
    result = x - y
    subtract_time = (datetime.now() - start_time).microseconds
    # compressed subtract
    start_time = datetime.now()
    compressed_sum = compressed_x - compressed_y
    compressed_subtract_time = (datetime.now() - start_time).microseconds
    # multiply
    start_time = datetime.now()
    result = x * 3.14159
    multiply_time = (datetime.now() - start_time).microseconds
    # compressed multiply
    start_time = datetime.now()
    result = compressed_x * torch.pi
    compressed_multiply_time = (datetime.now() - start_time).microseconds
    start_time = datetime.now()
    decompressed = compressor.decompress(compressed_x)
    decompress_time = (datetime.now() - start_time).microseconds


if __name__ == "__main__":
    main()
