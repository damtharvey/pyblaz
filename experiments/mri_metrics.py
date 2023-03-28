import argparse
import pathlib
import itertools
import math

import torch
import torch.nn.functional as torchfunctional
import tqdm

import compression
import experiments.structural_similarity_time as ssim


MRI_DYNAMIC_RANGE = 1


class Channel:
    PRE_CONTRAST = 0
    FLAIR = 1
    POST_CONTRAST = 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--results", type=str, default="results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = pathlib.Path(args.data) / "lgg-mri-segmentation" / "as_tensors"
    example_paths = tuple(data_path.glob("*"))
    assert data_path.exists(), "Run tools/mri_tif_to_tensor.py first."
    results_path = pathlib.Path(args.results) / "mri"
    results_path.mkdir(parents=True, exist_ok=True)

    keep_proportion = 1

    to_write = ["original_shape,float_type,index_type,block_shape,keep_proportion,metric,error"]
    for float_type in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
        for index_type in (torch.int8, torch.int16):
            for block_shape in (4, 4, 4), (8, 8, 8), (16, 16, 16), (4, 8, 8), (4, 16, 16), (8, 16, 16):
                # n_coefficients = int(math.prod(block_shape) * keep_proportion)
                # mask = torch.zeros(block_shape, dtype=torch.bool)
                # for index in sorted(
                #     itertools.product(*(range(size) for size in block_shape)),
                #     key=lambda x: sum(x),
                # )[:n_coefficients]:
                #     mask[index] = True
                compressor = compression.Compressor(
                    block_shape,
                    dtype=float_type,
                    index_dtype=index_type,
                    # mask=mask,
                    device=device,
                )

                pretty_print_float_type = str(float_type)[6:]
                pretty_print_index_type = str(index_type)[6:]
                pretty_print_block_shape = "×".join(str(size) for size in block_shape)

                for example_index, example_path in tqdm.tqdm(
                    enumerate(example_paths),
                    total=len(example_paths),
                    desc=f"{str(float_type)[6:]} {str(index_type)[6:]} {pretty_print_block_shape}",
                ):
                    # According to the dataset README, some examples are missing channels.
                    # FLAIR is available in all examples.
                    flair = torch.load(example_path)[Channel.FLAIR].to(device)
                    compressed_flair = compressor.compress(flair)

                    pretty_print_original_shape = "×".join(str(x) for x in flair.shape)

                    to_write.append(
                        f"{pretty_print_original_shape},"
                        f"{pretty_print_float_type},"
                        f"{pretty_print_index_type},"
                        f"{pretty_print_block_shape},"
                        f"{keep_proportion},"
                        f"mean,"
                        f"{flair.mean() - compressed_flair.mean()}"
                    )
                    to_write.append(
                        f"{pretty_print_original_shape},"
                        f"{pretty_print_float_type},"
                        f"{pretty_print_index_type},"
                        f"{pretty_print_block_shape},"
                        f"{keep_proportion},"
                        f"variance,"
                        f"{flair.var(unbiased=False) - compressed_flair.variance()}"
                    )
                    to_write.append(
                        f"{pretty_print_original_shape},"
                        f"{pretty_print_float_type},"
                        f"{pretty_print_index_type},"
                        f"{pretty_print_block_shape},"
                        f"{keep_proportion},"
                        f"norm_2,"
                        f"{flair.norm(2) - compressed_flair.norm_2()}"
                    )

                    for other_example_index in range(example_index + 1, len(example_paths)):
                        other_flair = torch.load(example_paths[other_example_index])[Channel.FLAIR].to(device)

                        if flair.shape[0] < other_flair.shape[0]:
                            # Don't want to compress again.
                            other_flair = other_flair[: flair.shape[0]]
                        elif flair.shape[0] > other_flair.shape[0]:
                            other_flair = torchfunctional.pad(
                                other_flair, (0, 0, 0, 0, 0, flair.shape[0] - other_flair.shape[0])
                            )

                        other_compressed_flair = compressor.compress(other_flair)
                        other_pretty_print_original_shape = "×".join(str(x) for x in flair.shape)
                        to_write.append(
                            f"{pretty_print_original_shape},"
                            f"{other_pretty_print_original_shape}"
                            f"{pretty_print_float_type},"
                            f"{pretty_print_index_type},"
                            f"{pretty_print_block_shape},"
                            f"{keep_proportion},"
                            f"structural_similarity"
                            f"{ssim.structural_similarity(flair, other_flair, dynamic_range=MRI_DYNAMIC_RANGE) - compressed_flair.structural_similarity(other_compressed_flair, dynamic_range=MRI_DYNAMIC_RANGE)}"
                        )

    with open(results_path / "mri_metrics.csv", "w") as file:
        file.write("\n".join(to_write))


if __name__ == "__main__":
    main()
