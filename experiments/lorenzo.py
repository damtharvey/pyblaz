from pyblaz.compression import PyBlaz

import tqdm
import torch


def _test():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--dimensions", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=8, help="size of a hypercubic block")
    parser.add_argument("--max-size", type=int, default=2048)
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

        blocks_x = compressor.block(x)

        x1 = torch.zeros(blocks_x.shape)
        x2 = torch.zeros(blocks_x.shape)
        x3 = torch.zeros(blocks_x.shape)

        for k in range(blocks_x.shape[0]):
            for l in range(blocks_x.shape[1]):
                if l == 0:
                    x1[k, l] = blocks_x[k, l]
                else:
                    x1[k, l] = x1[k, l - 1]

                if l == 0 or l == 1:
                    x2[k, l] = blocks_x[k, l]
                else:
                    x2[k, l] = (2.0 * x2[k, l - 1]) - x2[k, l - 2]
                if l == 0 or l == 1 or l == 2:
                    x3[k, l] = blocks_x[k, l]
                else:
                    x3[k, l] = (3.0 * x3[k, l - 1]) - (3.0 * x3[k, l - 2]) + x3[k, l - 3]

        diff1 = blocks_x - x1
        diff2 = blocks_x - x2
        diff3 = blocks_x - x3
        predictor = []
        elemenated_block_numbers = []
        max_error = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
        for i in range(len(diff1)):
            list = [
                diff1[i, :].sum(),
                diff2[i, :].sum(),
                diff3[i, :].sum(),
            ]

            min_diff = min(list)

            min_index = list.index(min(list))
            if min_diff <= max_error:
                min_index = list.index(min(list))
                predictor.append(min_index + 1)
                elemenated_block_numbers.append(i)
        print("total number of blocks=", len(blocks_x))
        print("total number of blocks elemenated=", len(elemenated_block_numbers))
        print("Predictor used=", predictor)


if __name__ == "__main__":
    _test()
