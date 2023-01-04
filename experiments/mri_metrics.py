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

    data_path = pathlib.Path(args.data) / "lgg-mri-segmentation" / "as_tensors"
    assert data_path.exists(), "Run tools/mri_tif_to_tensor.py first."
    results_path = pathlib.Path(args.results) / "mri"
    results_path.mkdir(parents=True, exist_ok=True)

    compressor = compression.Compressor((8, 8, 8))

    to_write = [
        "uncompressed_mean,compressed_mean,"
        "uncompressed_variance,compressed_variance,"
        "uncompressed_norm2,compressed_norm2"
    ]
    for example_path in tqdm.tqdm(tuple(data_path.glob("*")), desc="Measuring error"):
        # According to the dataset README, some examples are missing channels. FLAIR is available in all examples.
        flair = torch.load(example_path)[Channel.FLAIR]
        compressed_flair = compressor.compress(flair)

        to_write.append(
            f"{flair.mean()},{compressed_flair.mean()},"
            f"{flair.var(unbiased=False)},{compressed_flair.variance()},"
            f"{flair.norm(2)},{compressed_flair.norm_2()}"
        )

    with open(results_path / "mri_metrics.csv", "w") as file:
        file.write("\n".join(to_write))


if __name__ == "__main__":
    main()
