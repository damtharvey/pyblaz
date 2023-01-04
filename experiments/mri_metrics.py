import argparse
import pathlib

import torch
import tqdm

import compression


class Channel:
    PRE_CONTRAST = 0
    FLAIR = 1
    POST_CONTRAST = 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../data")
    parser.add_argument("--results", type=str, default="../results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = pathlib.Path(args.data) / "lgg-mri-segmentation" / "as_tensors"
    assert data_path.exists(), "Run tools/mri_tif_to_tensor.py first."
    results_path = pathlib.Path(args.results) / "mri"
    results_path.mkdir(parents=True, exist_ok=True)

    to_write = ["float_type,index_type,block_shape,metric,error"]
    for float_type in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
        for index_type in (torch.int8, torch.int16):
            for block_shape in (4, 4, 4), (8, 8, 8), (16, 16, 16), (4, 8, 8), (4, 16, 16), (8, 16, 16):
                compressor = compression.Compressor(block_shape, dtype=float_type, index_dtype=index_type)
                pretty_print_float_type = str(float_type)[6:]
                pretty_print_index_type = str(index_type)[6:]
                pretty_print_block_shape = "×".join(str(size) for size in block_shape)

                for example_path in tqdm.tqdm(
                    tuple(data_path.glob("*")),
                    desc=f"{str(float_type)[6:]} {str(index_type)[6:]} {pretty_print_block_shape}",
                ):
                    # According to the dataset README, some examples are missing channels.
                    # FLAIR is available in all examples.
                    flair = torch.load(example_path)[Channel.FLAIR].to(device)
                    compressed_flair = compressor.compress(flair)

                    to_write.append(
                        f"{pretty_print_float_type},"
                        f"{pretty_print_index_type},"
                        f"{pretty_print_block_shape},mean,"
                        f"{flair.mean() - compressed_flair.mean()}"
                    )
                    to_write.append(
                        f"{pretty_print_float_type},"
                        f"{pretty_print_index_type},"
                        f"{pretty_print_block_shape},variance,"
                        f"{flair.var(unbiased=False) - compressed_flair.variance()}"
                    )
                    to_write.append(
                        f"{pretty_print_float_type},"
                        f"{pretty_print_index_type},"
                        f"{pretty_print_block_shape},norm_2,"
                        f"{flair.norm(2) - compressed_flair.norm_2()}"
                    )
    with open(results_path / "mri_metrics.csv", "w") as file:
        file.write("\n".join(to_write))


if __name__ == "__main__":
    main()
