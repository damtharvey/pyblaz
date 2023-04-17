from pyblaz.compression import PyBlaz

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

    block_shape = (args.block_size,) * (args.dimensions - 1)
    print(block_shape)
    compressor = PyBlaz(
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

        # compress
        mean_subtraction = 0
        compressed_mean_subtraction = 0
        for i in range(0, x.size()[0]):
            compressed_x = compressor.compress(x[i])
            compressed_y = compressor.compress(y[i])
            mean_subtraction += abs(x[i].mean() - y[i].mean())
            compressed_mean_subtraction += abs(compressed_y.mean() - compressed_x.mean())
        difference = abs(mean_subtraction - compressed_mean_subtraction)

        results.append(torch.mean(mean_subtraction))
        results.append(torch.mean(compressed_mean_subtraction))
        results.append(torch.mean(difference))
        table.append(results)
    print(
        tabulate(
            table,
            headers=(
                "size",
                "mean difference",
                "compressed mean difference",
                "difference between mean and compressed mean",
            ),
        )
    )


if __name__ == "__main__":
    _test()
