import math
import pathlib

import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import pandas as pd


def main():
    save_path = pathlib.Path("results/time/plots")
    save_path.mkdir(parents=True, exist_ok=True)
    colors = list(matplotlib.colors.TABLEAU_COLORS.keys())
    line_styles = ("dotted", "dashed", "solid")

    dataframe = pd.read_csv(f"results/time/results.csv")
    dataframe = dataframe.replace("OOM", np.NAN)

    # operations = "compress,negate,add,multiply,dot,norm2,mean,variance,cosine_similarity,structural_similarity,decompress".split(",")
    operations = ("mean", "variance", "norm2", "cosine_similarity", "structural_similarity")

    for operation in operations:
        plt.clf()
        for color_index, dimensions in enumerate(dataframe["dimensions"].unique()):
            for line_style, dtype in zip(line_styles, dataframe["dtype"].unique()):
                for block_size in dataframe["block_size"].unique():
                    subframe = dataframe[
                        (dataframe["dimensions"] == dimensions)
                        & (dataframe["dtype"] == dtype)
                        & (dataframe["block_size"] == block_size)
                    ]
                    horizontal_values = [str(x) for x in sorted(subframe["size"].unique())]

                    plt.plot(
                        horizontal_values,
                        subframe[operation].astype(float) / 10**6,
                        color=colors[color_index],
                        linestyle=line_style,
                        # label=f"BS{block_size}, {dimensions}D",
                    )
        plt.title(f"{operation.replace('_', ' ')} time".title())
        plt.xlabel("array size")
        plt.xticks(rotation=-30)
        plt.ylabel("time (seconds)")
        plt.yscale("log")
        # plt.legend()
        plt.tight_layout()
        # plt.savefig(save_path / f"time_{operation}.pdf")
        plt.show()


if __name__ == "__main__":
    main()
