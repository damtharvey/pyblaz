import math
import pathlib

import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import pandas as pd


def main():
    save_path = pathlib.Path("results/time_ssim/plots")
    save_path.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_csv(f"results/time_ssim/results.csv")
    dataframe = dataframe.replace("OOM", np.NAN)

    operations = ("uncompressed_structural_similarity", "compress", "compressed_structural_similarity", "decompress")

    for dimensions in dataframe["dimensions"].unique():
        for dtype in dataframe["dtype"].unique():
            for block_size in dataframe["block_size"].unique():
                plt.clf()
                for operation in operations:
                    subframe = dataframe[
                        (dataframe["dimensions"] == dimensions)
                        & (dataframe["dtype"] == dtype)
                        & (dataframe["block_size"] == block_size)
                    ]
                    horizontal_values = [str(x) for x in sorted(subframe["size"].unique())]

                    plt.plot(
                        horizontal_values,
                        subframe[operation].astype(float),
                        label=f"{operation.replace('_', ' ')}",
                    )
                plt.title(
                    f"Uncompressed vs. Compressed Structural Similarity Time\n"
                    f"{dimensions}D, {dtype}, {'×'.join((str(block_size),) * dimensions)} blocks"
                )
                plt.xlabel("array size")
                plt.xticks(rotation=-30)
                plt.ylabel("time (seconds)")
                plt.yscale("log")
                plt.legend()
                plt.tight_layout()
                plt.savefig(save_path / f"{dimensions}d_{dtype}_{'x'.join((str(block_size),) * dimensions)}.pdf")
                # plt.show()


if __name__ == "__main__":
    main()
