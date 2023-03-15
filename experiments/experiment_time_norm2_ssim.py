import argparse
import pathlib
import math
import itertools

import tqdm

import compression
import torch
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--dimensions", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=8, help="size of a hypercubic block")
    parser.add_argument("--max-size", type=int, default=1 << 10)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
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
        default="int8",
        choices=(
            index_dtypes := {"int8": torch.int8, "int16": torch.int16, "int32": torch.int32, "int64": torch.int64}
        ),
    )
    parser.add_argument("--keep-proportion", type=float, default=0.5)
    parser.add_argument("--results-path", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default="time_norm2_ssim")
    args = parser.parse_args()

    dtype = dtypes[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_save_path = pathlib.Path(args.results_path) / args.experiment_name / f"{args.dimensions}d"
    results_save_path.mkdir(parents=True, exist_ok=True)

    block_shape = (args.block_size,) * args.dimensions
    n_coefficients = int(math.prod(block_shape) * args.keep_proportion)
    mask = torch.zeros(block_shape, dtype=torch.bool)
    for index in sorted(
        itertools.product(*(range(size) for size in block_shape)),
        key=lambda coordinates: sum(coordinates),
    )[:n_coefficients]:
        mask[index] = True
    compressor = compression.Compressor(
        block_shape=block_shape,
        dtype=dtype,
        index_dtype=index_dtypes[args.index_dtype],
        mask=mask,
        device=device,
    )

    to_write = ["size,compress,norm2,ssim,compressed_norm2,compressed_ssim,decompress"]

    for size in tqdm.tqdm(
        tuple(1 << p for p in range(args.block_size.bit_length() - 1, args.max_size.bit_length())),
        desc=f"{args.dimensions}D",
    ):
        results = []
        for run_number in range(args.runs + 1):
            x = torch.rand((size,) * args.dimensions, dtype=dtype, device=device)
            y = torch.rand((size,) * args.dimensions, dtype=dtype, device=device)

            # compress
            start_time = datetime.now()
            compressed_x = compressor.compress(x)
            compressed_y = compressor.compress(y)
            compress = ((datetime.now() - start_time) / 2).microseconds

            # norm2
            start_time = datetime.now()
            _ = (x - y).norm(2)
            norm2 = (datetime.now() - start_time).microseconds

            dynamic_range = max(x.max(), y.max()) - min(x.min(), y.min())
            # ssim
            start_time = datetime.now()
            _ = structural_similarity(x, y, dynamic_range=dynamic_range)
            ssim = (datetime.now() - start_time).microseconds

            # c norm2
            start_time = datetime.now()
            _ = compressed_x.norm_2()
            compressed_norm2 = (datetime.now() - start_time).microseconds

            # c ssim
            start_time = datetime.now()
            _ = compressed_x.structural_similarity(compressed_y, dynamic_range=dynamic_range)
            compressed_ssim = (datetime.now() - start_time).microseconds

            # decompression
            start_time = datetime.now()
            _ = compressor.decompress(compressed_x)
            decompress = (datetime.now() - start_time).microseconds

            if run_number > 0:
                results.append(
                    [
                        size,
                        compress,
                        norm2,
                        ssim,
                        compressed_norm2,
                        compressed_ssim,
                        decompress,
                    ]
                )

        results = torch.tensor(results, dtype=torch.float64).mean(0).round()
        to_write.append(",".join(str(int(number)) for number in results))

    with open(
        results_save_path / f"bs{args.block_size}_{str(dtype)[6:]}.csv",
        "w",
    ) as file:
        file.write("\n".join(to_write))


def structural_similarity(
    x,
    y,
    luminance_weight: float = 1,
    contrast_weight: float = 1,
    structure_weight: float = 1,
    dynamic_range: float = 0,
    luminance_stabilization: float = 0.01,
    contrast_stabilization: float = 0.03,
) -> float:
    x_mean = x.mean()
    y_mean = y.mean()
    x_variance = x.var()
    y_variance = y.var()
    covariance_ = covariance(x, y)

    luminance_stabilizer = luminance_stabilization * dynamic_range
    contrast_stabilizer = contrast_stabilization * dynamic_range
    similarity_stabilizer = contrast_stabilizer / 2

    luminance_similarity = (2 * x_mean * y_mean + luminance_stabilizer) / (
        x_mean**2 + y_mean**2 + luminance_stabilizer
    )
    contrast_similarity = (2 * x_variance * y_variance + contrast_stabilizer) / (
        x_variance**2 + y_variance**2 + contrast_stabilizer
    )
    structure_similarity = (covariance_ + similarity_stabilizer) / (x_variance * y_variance + similarity_stabilizer)

    return (
        luminance_similarity**luminance_weight
        + contrast_similarity**contrast_weight
        + structure_similarity**structure_weight
    )


def covariance(x, y):
    return ((x - x.mean()) * (y - y.mean())).mean()


if __name__ == "__main__":
    main()
