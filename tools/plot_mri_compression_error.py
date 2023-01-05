import argparse
import pathlib

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="../results")
    args = parser.parse_args()

    results_path = pathlib.Path(args.results) / "mri"
    save_path = results_path / "figures"
    save_path.mkdir(parents=True, exist_ok=True)

    flair_mean = 0.087

    colors = list(matplotlib.colors.TABLEAU_COLORS.keys())
    index_types = ("int8", "int16")
    index_type_markers = ("s", "D")
    index_type_offsets = (-0.038, 0.038)
    block_shapes = (4, 4, 4), (8, 8, 8), (16, 16, 16), (4, 8, 8), (4, 16, 16), (8, 16, 16)
    block_shape_offsets = ((range_shapes := np.arange(len(block_shapes))) - range_shapes.mean()) * 0.16
    float_types = ("float16", "bfloat16", "float32", "float64")
    horizontal_values = np.arange(len(float_types))
    with open(results_path / "mri_metrics.csv") as file:
        dataframe = pd.read_csv(file)
    for metric in ("mean", "variance", "norm_2"):
        plt.clf()
        fig = plt.figure(figsize=(8, 4))
        ax1 = fig.add_subplot(111)

        for index_type, marker, index_type_offset in zip(index_types, index_type_markers, index_type_offsets):
            for block_shape, color, block_shape_offset in zip(block_shapes, colors, block_shape_offsets):
                error_means = []
                for center_position, float_type in enumerate(float_types):
                    selected_absolute_error = dataframe[
                        (dataframe.index_type == index_type)
                        & (dataframe.block_shape == "×".join(str(size) for size in block_shape))
                        & (dataframe.float_type == float_type)
                        & (dataframe.metric == metric)
                    ].error.abs()
                    if selected_absolute_error.hasnans:
                        error_means.append(float("nan"))
                    else:
                        error_means.append(selected_absolute_error.mean())
                        no_nans = selected_absolute_error.dropna()
                        #  norm_2 using float16 has one dot at about 14. Excluding it to make the plot more useful.
                        no_nans = no_nans[no_nans < 10]
                        ax1.scatter(
                            [center_position + block_shape_offset + index_type_offset] * len(no_nans),
                            no_nans,
                            s=4,
                            color=color,
                            alpha=0.05,
                        )

                ax1.scatter(
                    horizontal_values + block_shape_offset + index_type_offset, error_means, marker=marker, color=color
                )

        ax1.set_xticks(horizontal_values, float_types)
        legend = []
        for index_type, marker in zip(index_types, index_type_markers):
            legend.append(
                Line2D([0], [0], marker=marker, linestyle="", color="black", label=f"index type {index_type}")
            )
        for block_shape, color in zip(block_shapes, colors):
            legend.append(Patch(facecolor=color, label=f"{'×'.join(str(size) for size in block_shape)} blocks"))
        ax1.legend(handles=legend)
        ax1.set_title(f"Error between compressed and uncompressed {metric}")
        ax1.set_ylabel("absolute error")
        ax1.set_xlabel("floating-point type")

        ax2 = ax1.twinx()
        ax2.set_ylabel("relative error")
        ax2.set_ylim(ax1.get_ylim()[0] / flair_mean, ax1.get_ylim()[1] / flair_mean)

        fig.tight_layout()
        plt.savefig(save_path / f"mri_flair_{metric}_error.pdf")
        # plt.show()


if __name__ == "__main__":
    main()
