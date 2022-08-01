import argparse

import compression
import torch
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=4, help="size of a hypercubic block")
    parser.add_argument(
        "--index-dtype",
        type=str,
        default="int8",
        choices=(
            index_dtypes := {"int8": torch.int8, "int16": torch.int16, "int32": torch.int32, "int64": torch.int64}
        ),
    )
    args = parser.parse_args()

    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    compressor = compression.Compressor(
        block_shape=(args.block_size,) * args.dimensions,
        dtype=dtype,
        index_dtype=index_dtypes[args.index_dtype],
        device=device,
    )
    x = torch.randn(8192, 8192, dtype=dtype, device=device)
    y = torch.randn(8192, 8192, dtype=dtype, device=device)
    # x = torch.tensor([[0.1 * i * j for j in range(16)] for i in range(16)], dtype=dtype, device=device)
    # y = torch.tensor([[0.2 * i * j for j in range(16)] for i in range(16)], dtype=dtype, device=device)

    compressed_x = compressor.compress(x)
    compressed_y = compressor.compress(y)
    start_time = datetime.now()
    compressed_sum = compressed_x - compressed_y
    print(f"subtract operation time: {(datetime.now() - start_time).microseconds}")

    start_time = datetime.now()
    compressed_product = compressed_x * torch.pi
    print(f"scalar multiply operation time: {(datetime.now() - start_time).microseconds}")

    decompressed_sum = compressor.decompress(compressed_sum)
    uncompressed_sum = x - y
    print(f"subtract difference: {(uncompressed_sum - decompressed_sum).norm(torch.inf)}")

    decompressed_product = compressor.decompress(compressed_product)
    uncompressed_product = x * torch.pi
    print(f"scalar multiply difference: {(uncompressed_product - decompressed_product).norm(torch.inf)}")


if __name__ == "__main__":
    main()
