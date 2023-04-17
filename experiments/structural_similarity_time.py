import argparse
import pathlib
import math
import itertools

import tqdm

from pyblaz import compression
import torch
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument(
        "--index-dtype",
        type=str,
        default="int16",
        choices=(
            index_dtypes := {"int8": torch.int8, "int16": torch.int16, "int32": torch.int32, "int64": torch.int64}
        ),
    )
    parser.add_argument("--keep-proportion", type=float, default=0.5)
    parser.add_argument("--results-path", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default="time_ssim")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_save_path = pathlib.Path(args.results_path) / args.experiment_name
    results_save_path.mkdir(parents=True, exist_ok=True)

    dtypes = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }

    to_write = [
        "dimensions,dtype,size,block_size,uncompressed_structural_similarity,compress,compressed_structural_similarity,decompress"
    ]

    for dtype_str, dtype in dtypes.items():
        for dimensions in range(2, 5):
            for block_size in (4, 8, 16):
                block_shape = (block_size,) * dimensions
                n_coefficients = int(math.prod(block_shape) * args.keep_proportion)
                mask = torch.zeros(block_shape, dtype=torch.bool)
                for index in sorted(
                    itertools.product(*(range(size) for size in block_shape)),
                    key=lambda coordinates: sum(coordinates),
                )[:n_coefficients]:
                    mask[index] = True
                compressor = compression.PyBlaz(
                    block_shape=block_shape,
                    dtype=dtype,
                    index_dtype=index_dtypes[args.index_dtype],
                    mask=mask,
                    device=device,
                )

                for size in tqdm.tqdm(
                    tuple(1 << p for p in range(block_size.bit_length() - 1, args.max_size.bit_length())),
                    desc=f"{dtype_str} {dimensions}D bs{block_size}",
                ):
                    try:
                        results = []
                        for run_number in range(args.runs + 1):
                            x = torch.rand((size,) * dimensions, dtype=dtype, device=device)
                            y = torch.rand((size,) * dimensions, dtype=dtype, device=device)

                            # uncompressed ssim
                            start_time = datetime.now()
                            _ = structural_similarity(x, y)
                            ssim = (datetime.now() - start_time).microseconds

                            # compress
                            start_time = datetime.now()
                            compressed_x = compressor.compress(x)
                            compressed_y = compressor.compress(y)
                            compress = ((datetime.now() - start_time) / 2).microseconds

                            # compressed structural similarity
                            start_time = datetime.now()
                            _ = compressed_x.structural_similarity(compressed_y)
                            compressed_structural_similarity = (datetime.now() - start_time).microseconds

                            # decompression
                            start_time = datetime.now()
                            _ = compressor.decompress(compressed_x)
                            decompress = (datetime.now() - start_time).microseconds

                            if run_number > 0:
                                results.append(
                                    [
                                        ssim,
                                        compress,
                                        compressed_structural_similarity,
                                        decompress,
                                    ]
                                )

                        results = torch.tensor(results, dtype=torch.float64).mean(0).round()
                        to_write.append(
                            f"{dimensions},{dtype_str},{size},{block_size},"
                            + ",".join(str(int(number)) for number in results)
                        )
                    except torch.cuda.OutOfMemoryError:
                        to_write.append(
                            f"{dimensions},{dtype_str},{size},{block_size}," + ",".join("OOM" for _ in range(4))
                        )

    with open(results_save_path / "results.csv", "w") as file:
        file.write("\n".join(to_write))


def covariance(x, y):
    return ((x - x.mean()) * (y - y.mean())).mean()


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
    x_standard_deviation = x_variance**0.5
    y_standard_deviation = y_variance**0.5
    covariance_ = covariance(x, y)

    luminance_stabilizer = luminance_stabilization * dynamic_range
    contrast_stabilizer = contrast_stabilization * dynamic_range
    similarity_stabilizer = contrast_stabilizer / 2

    luminance_similarity = (2 * x_mean * y_mean + luminance_stabilizer) / (
        x_mean**2 + y_mean**2 + luminance_stabilizer
    )
    contrast_similarity = (2 * x_standard_deviation * y_standard_deviation + contrast_stabilizer) / (
        x_variance + y_variance + contrast_stabilizer
    )
    structure_similarity = (covariance_ + similarity_stabilizer) / (
        x_standard_deviation * y_standard_deviation + similarity_stabilizer
    )

    return (
        luminance_similarity**luminance_weight
        * contrast_similarity**contrast_weight
        * structure_similarity**structure_weight
    )


if __name__ == "__main__":
    main()
