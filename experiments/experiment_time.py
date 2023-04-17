import argparse
import pathlib
import math
import itertools
import time

import tqdm

from pyblaz import compression
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-size", type=int, default=8192)
    parser.add_argument("--keep-proportion", type=float, default=0.5)
    parser.add_argument("--results-path", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default="time")
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

    index_dtypes = {
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
    }

    to_write = [
        "dimensions,dtype,index_dtype,size,block_size,compress,negate,add,multiply,dot,norm2,mean,variance,cosine_similarity,structural_similarity,decompress"
    ]

    for dtype_str, dtype in dtypes.items():
        for index_dtype_str, index_dtype in index_dtypes.items():
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
                        index_dtype=index_dtype,
                        mask=mask,
                        device=device,
                    )

                    for size in tqdm.tqdm(
                        tuple(1 << p for p in range(block_size.bit_length() - 1, args.max_size.bit_length())),
                        desc=f"{dtype_str} {index_dtype_str} {dimensions}D bs{block_size}",
                    ):
                        try:
                            results = []
                            for run_number in range(args.runs + 1):
                                x = torch.rand((size,) * dimensions, dtype=dtype, device=device)
                                y = torch.rand((size,) * dimensions, dtype=dtype, device=device)

                                # compress
                                start_time = time.time()
                                compressed_x = compressor.compress(x)
                                compressed_y = compressor.compress(y)
                                compress = (time.time() - start_time) / 2

                                # compressed negate
                                start_time = time.time()
                                _ = -compressed_x
                                compressed_negate = time.time() - start_time

                                # compressed add
                                start_time = time.time()
                                _ = compressed_x + compressed_y
                                compressed_add = time.time() - start_time

                                # compressed multiply
                                start_time = time.time()
                                _ = compressed_x * 3.14159
                                compressed_multiply = time.time() - start_time

                                # compressed dot
                                start_time = time.time()
                                _ = compressed_x.dot(compressed_y)
                                compressed_dot = time.time() - start_time

                                # compressed norm2
                                start_time = time.time()
                                _ = compressed_x.norm_2()
                                compressed_norm2 = time.time() - start_time

                                # compressed mean
                                start_time = time.time()
                                _ = compressed_x.mean()
                                compressed_mean = time.time() - start_time

                                # compressed variance
                                start_time = time.time()
                                _ = compressed_x.variance()
                                compressed_variance = time.time() - start_time

                                # compressed cosine similarity
                                start_time = time.time()
                                _ = compressed_x.cosine_similarity(compressed_y)
                                compressed_cosine_similarity = time.time() - start_time

                                # compressed structural similarity
                                start_time = time.time()
                                _ = compressed_x.structural_similarity(compressed_y)
                                compressed_structural_similarity = time.time() - start_time

                                # decompression
                                start_time = time.time()
                                _ = compressor.decompress(compressed_x)
                                decompress = time.time() - start_time

                                if run_number > 0:
                                    results.append(
                                        [
                                            compress,
                                            compressed_negate,
                                            compressed_add,
                                            compressed_multiply,
                                            compressed_dot,
                                            compressed_norm2,
                                            compressed_mean,
                                            compressed_variance,
                                            compressed_cosine_similarity,
                                            compressed_structural_similarity,
                                            decompress,
                                        ]
                                    )

                            results = torch.tensor(results, dtype=torch.float64).mean(0)
                            to_write.append(
                                f"{dimensions},{dtype_str},{index_dtype_str},{size},{block_size},"
                                + ",".join(str(float(number)) for number in results)
                            )
                        except torch.cuda.OutOfMemoryError:
                            to_write.append(
                                f"{dimensions},{dtype_str},{index_dtype_str},{size},{block_size},"
                                + ",".join("OOM" for _ in range(11))
                            )

    with open(results_save_path / "results.csv", "w") as file:
        file.write("\n".join(to_write))


if __name__ == "__main__":
    main()
