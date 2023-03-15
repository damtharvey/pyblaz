import math
import pathlib

import matplotlib.pyplot as plt
import matplotlib.colors
import pandas as pd


def main():
    save_path = pathlib.Path("results/time_norm2_ssim/plots")
    save_path.mkdir(parents=True, exist_ok=True)

    # colors = list(matplotlib.colors.TABLEAU_COLORS.keys())

    dataframe = pd.read_csv(f"results/time_norm2_ssim/3d/bs8_bfloat16.csv")
    horizontal_values = [str(x) for x in dataframe["size"]]



    for dimensions in (3,):
        for operation in ("norm2", "ssim"):
            plt.clf()
            plt.plot(
                horizontal_values,
                dataframe[f"compressed_{operation}"] / 10**6,
                label=f"compressed {operation}",
            )
            plt.plot(
                horizontal_values,
                dataframe["decompress"] / 10**6,
                label=f"decompression",
            )
            plt.plot(
                horizontal_values,
                dataframe[operation] / 10**6,
                label=f"uncompressed {operation}",
            )
            plt.plot(
                horizontal_values,
                dataframe["compress"] / 10**6,
                label=f"compression",
            )

            plt.title(f"{dimensions}d {operation}")
            plt.xlabel("array size")
            plt.xticks(rotation=-30)
            plt.ylabel("time (seconds)")
            plt.yscale("log")
            plt.legend()
            plt.tight_layout()
            plt.savefig(save_path / f"{dimensions}d_{operation}.pdf")
            # plt.show()


if __name__ == "__main__":
    main()
