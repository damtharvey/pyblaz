from compression import Compressor

import tqdm
import torch


def _test():
    import argparse
    from tabulate import tabulate

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--dimensions", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=8, help="size of a hypercubic block")
    parser.add_argument("--max-size", type=int, default=256)
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

    block_shape = (args.block_size,) * (args.dimensions)

    compressor = Compressor(
        block_shape=block_shape,
        dtype=dtype,
        index_dtype=index_dtypes[args.index_dtype],
        device=device,
    )
    table = []
    for size in tqdm.tqdm(
        tuple(1 << p for p in range(args.block_size.bit_length() - 1, args.max_size.bit_length())),
        desc=f"time {args.dimensions}D",
    ):
        results = [size]
        x = torch.randn((size,) * args.dimensions, dtype=dtype, device=device)
        y = torch.randn((size,) * args.dimensions, dtype=dtype, device=device)
        compressed_x = compressor.compress(x)
        compressed_y = compressor.compress(y)
        diff = (((x - y) ** 2).sum()) ** (0.5)
        compress_diff = ((compressor.decompress(compressed_x - compressed_y) ** 2).sum()) ** (0.5)

        compressed_x_mean = compressed_x.mean_blockwise()
        compressed_y_mean = compressed_y.mean_blockwise()

        size_x = compressed_x_mean.size()
        product_x = size_x[0] * size_x[1] * size_x[2]
        size_y = compressed_y_mean.size()
        product_y = size_y[0] * size_y[1] * size_y[2]

        compress_diff_mean = (
            ((compressed_x_mean.sum() / product_x - compressed_y_mean.sum() / product_y) ** 2).sum()
        ) ** (0.5)
        diff_mean = (((x.mean() - y.mean()) ** 2).sum()) ** (0.5)

        results.append(diff)
        results.append(compress_diff)

        results.append(diff_mean)
        results.append(compress_diff_mean)
        table.append(results)
    print(
        tabulate(
            table,
            headers=(
                "size",
                "wass diff",
                "wass compress_diff",
                "wass diff_mean",
                "wass compress_diff_mean",
            ),
        )
    )


if __name__ == "__main__":
    _test()
