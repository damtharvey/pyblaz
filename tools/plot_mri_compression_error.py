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

    colors = list(matplotlib.colors.TABLEAU_COLORS.keys())
    index_types = ("int8", "int16")
    index_type_markers = ("s", "D")
    index_type_offsets = (-0.05, 0.05)
    block_sizes = (4, 8, 16)
    block_size_offsets = (-0.2, 0, 0.2)
    float_types = ("float16", "bfloat16", "float32", "float64")
    horizontal_values = np.array(range(len(float_types)))

    with open(results_path / f"mri_metrics.csv") as file:
        dataframe = pd.read_csv(file)

    for metric in ("mean", "variance", "norm_2"):
        plt.clf()
        for index_type, marker, index_type_offset in zip(index_types, index_type_markers, index_type_offsets):
            for block_size, color, block_type_offset in zip(block_sizes, colors, block_size_offsets):
                error_means = []
                for center_position, float_type in enumerate(float_types):
                    selected_absolute_error = dataframe[
                        (dataframe.index_type == index_type)
                        & (dataframe.block_size == block_size)
                        & (dataframe.float_type == float_type)
                        & (dataframe.metric == metric)
                    ].error.abs()
                    if selected_absolute_error.hasnans:
                        error_means.append(float("nan"))
                    else:
                        error_means.append(selected_absolute_error.mean())
                        no_nans = selected_absolute_error.dropna()
                        plt.scatter(
                            [center_position + block_type_offset + index_type_offset] * len(no_nans),
                            no_nans,
                            s=8,
                            color=color,
                            alpha=0.05,
                        )

                plt.scatter(
                    horizontal_values + block_type_offset + index_type_offset, error_means, marker=marker, color=color
                )

        plt.xticks(horizontal_values, float_types)
        legend = []
        for index_type, marker in zip(index_types, index_type_markers):
            legend.append(
                Line2D([0], [0], marker=marker, linestyle="", color="black", label=f"index type {index_type}")
            )
        for block_size, color in zip(block_sizes, colors):
            legend.append(Patch(facecolor=color, label=f"block size {block_size}"))
        plt.legend(handles=legend)
        plt.title(f"Error between compressed and uncompressed {metric} of FLAIR")
        plt.ylabel("absolute error")
        plt.xlabel("floating-point type")
        plt.tight_layout()
        plt.savefig(save_path / f"mri_flair_{metric}_error.pdf")
        plt.savefig(save_path / f"mri_flair_{metric}_error.png", dpi=600)
        # plt.show()


if __name__ == "__main__":
    main()
