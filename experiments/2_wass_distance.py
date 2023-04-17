# As described https://bioc.ism.ac.jp/packages/3.10/bioc/vignettes/waddR/inst/doc/wasserstein_metric.html


from pyblaz.compression import PyBlaz

import tqdm
import torch
import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


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

        compressed_x = compressor.compress(x)
        compressed_y = compressor.compress(y)

        compressed_x_mean = compressed_x.mean_blockwise()
        compressed_y_mean = compressed_y.mean_blockwise()

        compressed_x_variance = compressed_x.standard_deviation_blockwise()
        compressed_y_variance = compressed_y.standard_deviation_blockwise()

        # flatten

        softmax_compressed_x_mean = softmax(np.asarray(compressed_x_mean))
        softmax_compressed_y_mean = softmax(np.asarray(compressed_y_mean))

        softmax_compressed_x_variance = softmax(np.asarray(compressed_x_variance))
        softmax_compressed_y_variance = softmax(np.asarray(compressed_y_variance))

        softmax_x = softmax(np.asarray(x))
        softmax_y = softmax(np.asarray(y))

        sorted_softmax_compressed_x_mean = np.sort(softmax_compressed_x_mean)
        sorted_softmax_compressed_y_mean = np.sort(softmax_compressed_y_mean)

        sorted_softmax_compressed_x_variance = np.sort(softmax_compressed_x_variance)
        sorted_softmax_compressed_y_variance = np.sort(softmax_compressed_y_variance)

        sorted_softmax_x = np.sort(softmax_x)
        sorted_softmax_y = np.sort(softmax_y)

        wass_distance_mean = [
            (a - b) ** 2 for a, b in zip(sorted_softmax_compressed_x_mean, sorted_softmax_compressed_y_mean)
        ]

        wass_distance_varaince = [
            (a - b) ** 2 for a, b in zip(sorted_softmax_compressed_x_variance, sorted_softmax_compressed_y_variance)
        ]

        mean_compressed_x = softmax_compressed_x_mean.mean()
        mean_compressed_y = softmax_compressed_y_mean.mean()
        numerator = [
            (a - mean_compressed_x) * (b - mean_compressed_y)
            for a, b in zip(sorted_softmax_compressed_x_mean, sorted_softmax_compressed_y_mean)
        ]
        denominator1 = [(a - mean_compressed_x) ** 2 for a in sorted_softmax_compressed_x_mean]
        denominator2 = [(b - mean_compressed_y) ** 2 for b in sorted_softmax_compressed_y_mean]
        denominator = (sum(denominator1) * sum(denominator2)) ** (1 / 2)
        p_value = sum(numerator) / denominator
        wass_distance_pearson = (
            2 * sorted_softmax_compressed_x_variance * sorted_softmax_compressed_y_variance * (1 - p_value)
        )

        wass_distance_compressed = (
            np.mean(wass_distance_mean) + np.mean(wass_distance_varaince) + np.mean(wass_distance_pearson)
        )
        print(wass_distance_compressed)

        order = 3
        wass_distance = [
            (((abs(a - b)) ** order) ** (1 / order)).mean() for a, b in zip(sorted_softmax_x, sorted_softmax_y)
        ]
        diff = np.mean(wass_distance)

        # diff_mean = (((x.mean() - y.mean()) ** 2).sum()) ** (0.5)

        results.append(diff)
        # results.append(compress_diff)

        # results.append(diff_mean)
        results.append(wass_distance_compressed)
        table.append(results)
    print(
        tabulate(
            table,
            headers=(
                "size",
                "wass diff",
                "wass compress_diff blockwise mean",
            ),
        )
    )


if __name__ == "__main__":
    _test()
