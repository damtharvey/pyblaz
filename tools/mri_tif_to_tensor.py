import argparse
import pathlib

from PIL import Image
import torch
import torchvision.transforms.functional as transforms_f
import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../data")
    args = parser.parse_args()

    data_path = pathlib.Path(args.data)
    example_paths = data_path / "lgg-mri-segmentation" / "kaggle_3m"
    assert example_paths.exists(), (
        "Data can be downloaded from "
        "https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation and should be "
        "extracted like <data directory>/lgg-mri-segmentation/kaggle_3m/"
        "TCGA_CS_4941_19960909/TCGA_CS_4941_19960909_1.tif, etc."
    )
    save_path = data_path / "lgg-mri-segmentation" / "as_tensors"
    save_path.mkdir(parents=True, exist_ok=True)

    for example_path in tqdm.tqdm(tuple(example_paths.glob("*")), desc="Converting"):
        if example_path.is_dir():
            as_tensor = torch.stack(
                tuple(
                    transforms_f.to_tensor(Image.open(slice_path))
                    for slice_path in sorted(
                        (path for path in example_path.glob("*") if "mask" not in path.name),
                        key=lambda path: int(path.name.split("_")[-1][:-4]),
                    )
                )
            ).permute(
                1, 0, 2, 3  # depth, channels, width, height -> channels, depth, width, height
            )
            torch.save(as_tensor, save_path / f"{example_path.name}.pth")


if __name__ == "__main__":
    main()
