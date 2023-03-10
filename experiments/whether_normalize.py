import argparse
import pathlib

import torch
import tqdm

import compression


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimensions", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=8, help="size of a hypercubic block")
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
    parser.add_argument("--results-path", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default="whether_normalize/shallow_water")
    args = parser.parse_args()

    results_save_path = pathlib.Path(args.results_path) / args.experiment_name
    results_save_path.mkdir(parents=True, exist_ok=True)

    dtype = dtypes[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    compressor_normalize = compression.Compressor(
        block_shape=(args.block_size,) * args.dimensions,
        dtype=dtype,
        index_dtype=index_dtypes[args.index_dtype],
        do_differentiate=True,
        device=device,
    )
    compressor_no_normalize = compression.Compressor(
        block_shape=(args.block_size,) * args.dimensions,
        dtype=dtype,
        index_dtype=index_dtypes[args.index_dtype],
        do_differentiate=False,
        device=device,
    )

    data_directory = pathlib.Path("data") / "ShallowWatersEquations" / "output"

    to_write = [
        "normalize_l_inf,no_normalize_l_inf,normalize_l_2,no_normalize_l_2,normalize_l_1,no_normalize_l_1,normalize_l_0,no_normalize_l_0"
    ]

    for file_number in tqdm.tqdm(range(501)):
        file_name = f"{file_number}.txt"

        with open(data_directory / file_name) as file:
            uncompressed = torch.tensor(
                [[float(string) for string in line.split()] for line in file.readlines()], dtype=dtype, device=device
            )

        decompressed_normalize = compressor_normalize.decompress(compressor_normalize.compress(uncompressed))
        decompressed_no_normalize = compressor_no_normalize.decompress(compressor_no_normalize.compress(uncompressed))

        difference_normalize = decompressed_normalize - uncompressed
        difference_no_normalize = decompressed_no_normalize - uncompressed

        to_write.append(
            f"{difference_normalize.norm(torch.inf)},{difference_no_normalize.norm(torch.inf)},"
            f"{difference_normalize.norm(2)},{difference_no_normalize.norm(2)},"
            f"{difference_normalize.norm(1)},{difference_no_normalize.norm(1)},"
            f"{difference_normalize.norm(0)},{difference_no_normalize.norm(0)}"
        )

    with open(
        results_save_path
        / f"whether_normalize_{'x'.join(str(size) for size in compressor_normalize.block_shape)}_"
        f"{args.dtype}_"
        f"{args.index_dtype}.csv",
        "w",
    ) as file:
        file.write("\n".join(to_write))


if __name__ == "__main__":
    main()
