import argparse
import pathlib

import tqdm

from pyblaz import compression
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--dimensions", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=8, help="size of a hypercubic block")
    parser.add_argument("--max-size", type=int, default=256)
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
        default="int16",
        choices=(
            index_dtypes := {"int8": torch.int8, "int16": torch.int16, "int32": torch.int32, "int64": torch.int64}
        ),
    )
    parser.add_argument("--keep-proportion", type=float, default=0.5)
    parser.add_argument("--results-path", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default="ssim")
    args = parser.parse_args()

    dtype = dtypes[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_save_path = pathlib.Path(args.results_path) / args.experiment_name / f"{args.dimensions}d"
    results_save_path.mkdir(parents=True, exist_ok=True)

    block_shape = (args.block_size,) * args.dimensions
    # n_coefficients = int(math.prod(block_shape) * args.keep_proportion)
    # mask = torch.zeros(block_shape, dtype=torch.bool)
    # for index in sorted(
    #     itertools.product(*(range(size) for size in block_shape)),
    #     key=lambda coordinates: sum(coordinates),
    # )[:n_coefficients]:
    #     mask[index] = True
    compressor = compression.PyBlaz(
        block_shape=block_shape,
        dtype=dtype,
        index_dtype=index_dtypes[args.index_dtype],
        # mask=mask,
        device=device,
    )

    to_write = ["size,covariance,ssim"]

    for size in tqdm.tqdm(
        tuple(1 << p for p in range(args.block_size.bit_length() - 1, args.max_size.bit_length())),
        desc=f"{args.dimensions}D",
    ):
        x = (
            (10 * (torch.randn((size,) * args.dimensions) * torch.randn((1,)) + torch.randn((1,))))
            .type(dtype)
            .to(device)
        )
        y = (
            (10 * (torch.randn((size,) * args.dimensions) * torch.randn((1,)) + torch.randn((1,))))
            .type(dtype)
            .to(device)
        )

        dynamic_range = max(x.max(), y.max()) - min(x.min(), y.min())

        covariance_ = covariance(x, y)
        structural_similarity_ = structural_similarity(x, y, dynamic_range=dynamic_range)

        compressed_x = compressor.compress(x)
        compressed_y = compressor.compress(y)

        compressed_covariance = compressed_x.covariance(compressed_y)
        compressed_structural_similarity = compressed_x.structural_similarity(compressed_y, dynamic_range=dynamic_range)

        to_write.append(
            ",".join(
                [
                    str(size),
                    str(float((covariance_ - compressed_covariance).norm(torch.inf))),
                    str(float((structural_similarity_ - compressed_structural_similarity).norm(torch.inf))),
                ]
            )
        )

    with open(
        results_save_path / f"bs{args.block_size}_{str(dtype)[6:]}.csv",
        "w",
    ) as file:
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


if __name__ == "__main__":
    main()
