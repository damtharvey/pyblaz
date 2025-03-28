import argparse
import time
import tqdm
import torch
import itertools

from pyblaz.compression import PyBlaz


def benchmark_compression():
    """
    Benchmark the compression and operations on compressed tensors.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=8, help="size of a hypercubic block")
    parser.add_argument("--max-size", type=int, default=512)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=(
            dtypes := {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
                "float64": torch.float64,
            }
        ),
    )
    parser.add_argument(
        "--index-dtype",
        type=str,
        default="int16",
        choices=(index_dtypes := {"int8": torch.int8, "int16": torch.int16}),
    )
    args = parser.parse_args()

    dtype = dtypes[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    block_shape = (args.block_size,) * args.dimensions

    compressor = PyBlaz(
        block_shape=block_shape,
        dtype=dtype,
        index_dtype=index_dtypes[args.index_dtype],
        device=device,
    )

    time_table = []
    error_table = []
    headers = (
        "size",
        "codec",
        "negate",
        "add",
        "add_scalar",
        "multiply",
        "dot",
        "norm2",
        "mean",
        "variance",
        "cosine",
        "covariance",
    )

    for size in tqdm.tqdm(
        tuple(1 << p for p in range(args.block_size.bit_length() - 1, args.max_size.bit_length())),
        desc=f"{args.dimensions}D",
    ):
        size += 1
        time_results = [size]
        error_results = [size]

        x = torch.randn((size,) * args.dimensions, dtype=dtype, device=device) - 1
        y = torch.randn((size,) * args.dimensions, dtype=dtype, device=device) + 2

        # compress
        start_time = time.time()
        compressed_x = compressor.compress(x)
        compressed_y = compressor.compress(y)
        time_results.append(time.time() - start_time)

        # compressed negate
        start_time = time.time()
        r = -compressed_x
        time_results.append(time.time() - start_time)
        error_results.append((compressor.decompress(r) + x).norm(torch.inf))

        # compressed add
        start_time = time.time()
        r = compressed_x + compressed_y
        time_results.append(time.time() - start_time)
        error_results.append((compressor.decompress(r) - (x + y)).norm(torch.inf))

        # compressed add scalar
        start_time = time.time()
        r = compressed_x + 3.14159
        time_results.append(time.time() - start_time)
        error_results.append((compressor.decompress(r) - (x + 3.14159)).norm(torch.inf))

        # compressed multiply
        start_time = time.time()
        r = compressed_x * 3.14159
        time_results.append(time.time() - start_time)
        error_results.append((compressor.decompress(r) - (x * 3.14159)).norm(torch.inf))

        # compressed dot
        start_time = time.time()
        r = compressed_x.dot(compressed_y)
        time_results.append(time.time() - start_time)
        error_results.append(abs((r - (x * y).sum())))

        # compressed norm2
        start_time = time.time()
        r = compressed_x.norm_2()
        time_results.append(time.time() - start_time)
        error_results.append(abs(r - x.norm(2)))

        # compressed mean
        start_time = time.time()
        r = compressed_x.mean()
        time_results.append(time.time() - start_time)
        error_results.append(abs(r - x.mean()))

        # compressed variance
        start_time = time.time()
        r = compressed_x.variance()
        time_results.append(time.time() - start_time)
        error_results.append(abs(r - x.var()))

        # compressed cosine similarity
        start_time = time.time()
        r = compressed_x.cosine_similarity(compressed_y)
        time_results.append(time.time() - start_time)
        error_results.append(abs(r - torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(), 0)))

        # compressed covariance
        start_time = time.time()
        r = compressed_x.covariance(compressed_y)
        time_results.append(time.time() - start_time)
        error_results.append(abs(r - torch.cov(torch.stack([x.flatten(), y.flatten()]))[0, 1]))

        time_table.append(time_results)
        error_table.append(error_results)

    print("=" * 100)
    print("Time (s)")
    print("=" * 100)
    print(*headers, sep="\t")
    print("-" * 100)
    for row in time_table:
        print(*row, sep="\t")
    print("=" * 100)
    print("Error")
    print("=" * 100)
    print(*headers, sep="\t")
    print("-" * 100)
    for row in error_table:
        print(*row, sep="\t")
    print("=" * 100)


if __name__ == "__main__":
    benchmark_compression()
