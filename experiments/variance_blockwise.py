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

        compressed_x_mean = compressed_x.blockwise_variance()
        compressed_y_mean = compressed_y.blockwise_variance()
        # x_mean = x.mean_blockwise()
        size_x = compressed_x_mean.size()
        product_x = size_x[0] * size_x[1] * size_x[2]
        results.append(compressed_x_mean.size())
        results.append(x.var(unbiased=False))
        results.append(compressed_x_mean.sum() / product_x)

        size_y = compressed_y_mean.size()
        product_y = size_y[0] * size_y[1] * size_y[2]
        # results.append(compressed_y_mean.size())
        results.append(y.var(unbiased=False))
        results.append(compressed_y_mean.sum() / product_y)
        table.append(results)

    print(
        tabulate(
            table,
            headers=(
                "size",
                "mean arr size",
                "x.var()",
                "comp_x_var.sum()/prod",
                "y.var()",
                "comp_y_var.sum()/prod",
            ),
        )
    )


if __name__ == "__main__":
    _test()
