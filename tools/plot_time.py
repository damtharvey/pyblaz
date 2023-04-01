import math
import pathlib

import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def main():
    save_path = pathlib.Path(f"results/time/plots")
    save_path.mkdir(parents=True, exist_ok=True)

    colors = list(matplotlib.colors.TABLEAU_COLORS.keys())
    line_styles = ("dotted", "dashed", "solid")
    # markers = ("s", "D", "o")

    dataframe = pd.read_csv(f"results/time/results.csv")
    dataframe = dataframe.replace("OOM", np.NAN)

    operations = (
        "compress,negate,add,multiply,dot,norm2,mean,variance,cosine_similarity,structural_similarity,decompress".split(
            ","
        )
    )

    for operation in operations:
        for dimensions in dataframe["dimensions"].unique():
            for block_size in dataframe["block_size"].unique():
                plt.clf()
                figure = plt.figure(figsize=(5, 3.5))
                for color, dtype in zip(colors, dataframe["dtype"].unique()):
                    for line_style, index_dtype in zip(line_styles, dataframe["index_dtype"].unique()):
                        subframe = dataframe[
                            (dataframe["dimensions"] == dimensions)
                            & (dataframe["dtype"] == dtype)
                            & (dataframe["index_dtype"] == index_dtype)
                            & (dataframe["block_size"] == block_size)
                        ]
                        horizontal_values = [str(x) for x in sorted(subframe["size"].unique())]
                        plt.plot(
                            horizontal_values,
                            subframe[operation].astype(float),
                            # marker=marker,
                            color=color,
                            linestyle=line_style,
                            label=f"{dtype}, {index_dtype}",
                            figure=figure
                        )

                plt.title(
                    f"{operation.replace('_', ' ')} time\n".title()
                    + f"{dimensions}-dimensional arrays, block size {block_size}"
                )
                plt.xlabel("array size")
                plt.xticks(rotation=-30)
                plt.ylabel("time (seconds)")
                plt.yscale("log")
                # figure.legend(ncol=2)
                plt.tight_layout()
                plt.savefig(save_path / f"{dimensions}d_bs{block_size}_{operation}.pdf")
                # plt.show()
                plt.close()

    plt.clf()
    plt.figure(figsize=(3.5, 1.4))
    legend = []
    for color, dtype in zip(colors, dataframe["dtype"].unique()):
        for line_style, index_dtype in zip(line_styles, dataframe["index_dtype"].unique()):
            legend.append(Line2D([0], [0], linestyle=line_style, color=color, label=f"{dtype}, {index_dtype}"))
    plt.legend(loc="center", ncol=2, handles=legend)
    plt.gca().axis("off")
    plt.tight_layout()
    plt.savefig(save_path / "legend.pdf")
    # plt.show()
    plt.close()


if __name__ == "__main__":
    main()
