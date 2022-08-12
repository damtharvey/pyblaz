import argparse
import pathlib

import tqdm

import compression
import torch
from datetime import datetime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--dimensions", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=8, help="size of a hypercubic block")
    parser.add_argument("--max-size", type=int, default=256)
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
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
    parser.add_argument("--results-path", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default="time")
    args = parser.parse_args()

    dtype = dtypes[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_save_path = pathlib.Path(args.results_path) / args.experiment_name / f"{args.dimensions}d"
    results_save_path.mkdir(parents=True, exist_ok=True)

    compressor = compression.Compressor(
        block_shape=(args.block_size,) * args.dimensions,
        dtype=dtype,
        index_dtype=index_dtypes[args.index_dtype],
        device=device,
    )

    to_write = ["size,compress,negate,add,multiply,dot,norm2,mean,variance,cosine_similarity,decompress"]

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

            # compressed negate
            start_time = datetime.now()
            _ = -compressed_x
            compressed_negate = (datetime.now() - start_time).microseconds

            # compressed add
            start_time = datetime.now()
            _ = compressed_x + compressed_y
            compressed_add = (datetime.now() - start_time).microseconds

            # compressed multiply
            start_time = datetime.now()
            _ = compressed_x * 3.14159
            compressed_multiply = (datetime.now() - start_time).microseconds

            # compressed dot
            start_time = datetime.now()
            _ = compressed_x.dot(compressed_y)
            compressed_dot = (datetime.now() - start_time).microseconds

            # compressed norm2
            start_time = datetime.now()
            _ = compressed_x.norm_2()
            compressed_norm2 = (datetime.now() - start_time).microseconds

            # compressed mean
            start_time = datetime.now()
            _ = compressed_x.mean()
            compressed_mean = (datetime.now() - start_time).microseconds

            # compressed variance
            start_time = datetime.now()
            _ = compressed_x.variance()
            compressed_variance = (datetime.now() - start_time).microseconds

            # compressed cosine similarity
            start_time = datetime.now()
            _ = compressed_x.cosine_similarity(compressed_y)
            compressed_cosine_similarity = (datetime.now() - start_time).microseconds

            # decompression
            start_time = datetime.now()
            _ = compressor.decompress(compressed_x)
            decompress = (datetime.now() - start_time).microseconds

            if run_number > 0:
                results.append(
                    [
                        size,
                        compress,
                        compressed_negate,
                        compressed_add,
                        compressed_multiply,
                        compressed_dot,
                        compressed_norm2,
                        compressed_mean,
                        compressed_variance,
                        compressed_cosine_similarity,
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


if __name__ == "__main__":
    main()
