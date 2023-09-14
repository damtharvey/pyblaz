import argparse
import pathlib

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import tqdm

import plot_space


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results")
    args = parser.parse_args()

    results_path = pathlib.Path(args.results) / "mri"
    save_path = results_path / "figures"
    save_path.mkdir(parents=True, exist_ok=True)

    colors = list(matplotlib.colors.TABLEAU_COLORS.keys())
    block_shapes = (4, 4, 4), (8, 8, 8), (16, 16, 16), (4, 8, 8), (4, 16, 16), (8, 16, 16)
    index_types = {"int8": 8, "int16": 16}
    index_type_markers = ("s", "D")
    plot_errors(block_shapes, colors, index_type_markers, index_types, results_path, save_path)
    plot_legend(block_shapes, colors, index_type_markers, index_types, save_path)


def plot_errors(block_shapes, colors, index_type_markers, index_types, results_path, save_path):
    figure_shape = (4, 3.5)
    index_type_offsets = (-0.04, 0.04)
    block_shape_offsets = ((range_shapes := np.arange(len(block_shapes))) - range_shapes.mean()) * 0.165
    float_types = {
        # "float16": 16,
        # "bfloat16": 16,
        "float32": 32,
        "float64": 64,
    }
    horizontal_values = np.arange(len(float_types))

    with open(results_path / "mri_metrics.csv") as file:
        dataframe = pd.read_csv(file)

    for error in ("absolute_error", "relative_error"):
        for metric in ("mean", "variance", "norm_2", "structural_similarity"):
            with tqdm.tqdm(
                total=len(index_types) * len(block_shapes) * len(float_types), desc=f"{error} {metric}"
            ) as progress_bar:
                plt.clf()

                figure = plt.figure(figsize=figure_shape)
                axis = figure.add_subplot(111)
                axis.set_xticks(horizontal_values, float_types)
                ratio_axis = axis.twinx()
                ratio_axis.set_ylabel("compression ratio")

                max_error_without_nan = 0

                for index_type, marker, index_type_offset in zip(index_types, index_type_markers, index_type_offsets):
                    for block_shape, color, block_shape_offset in zip(block_shapes, colors, block_shape_offsets):
                        error_means = []
                        mean_compression_ratios = []
                        for center_position, float_type in enumerate(float_types):
                            selection = dataframe[
                                (dataframe.index_type == index_type)
                                & (dataframe.block_shape == "×".join(str(size) for size in block_shape))
                                & (dataframe.float_type == float_type)
                                & (dataframe.metric == metric)
                            ]
                            selected_error = selection[error].abs()
                            if selected_error.hasnans or any(selected_error == float("inf")):
                                error_means.append(float("nan"))
                            else:
                                max_error_without_nan = max(max_error_without_nan, selected_error.max())
                                error_means.append(selected_error.mean())

                            compression_ratios = []
                            for original_shape in selection.original_shape:
                                original_shape = tuple(int(x) for x in original_shape.split("×"))
                                compression_ratios.append(
                                    plot_space.uncompressed_size(original_shape)
                                    / plot_space.compressed_size(
                                        original_shape,
                                        block_shape,
                                        dataframe.keep_proportion,
                                        float_types[float_type],
                                        index_types[index_type],
                                    )
                                )

                            mean_compression_ratios.append(np.array(compression_ratios).mean())

                            axis.scatter(
                                [center_position + block_shape_offset + index_type_offset] * len(selected_error),
                                selected_error,
                                s=4,
                                color=color,
                                alpha=0.05 if metric != "structural_similarity" else 0.05**2,
                            )

                            progress_bar.update()

                        ratio_axis.bar(
                            horizontal_values + block_shape_offset + index_type_offset,
                            mean_compression_ratios,
                            color="black",
                            alpha=0.1,
                            width=0.01,
                        )

                        ratio_axis.scatter(
                            horizontal_values + block_shape_offset + index_type_offset,
                            mean_compression_ratios,
                            marker="_",
                            s=50,
                            color="black",
                        )

                        axis.scatter(
                            horizontal_values + block_shape_offset + index_type_offset,
                            error_means,
                            marker=marker,
                            color=color,
                            alpha=1,
                        )

                # axis.axvline(0.5, color="black")

                axis.set_title(f"{metric.replace('_', ' ')} {error.replace('_', ' ')}".title())
                axis.set_ylabel("error")
                axis.set_xlabel("floating-point type")
                axis.set_ylim(max(axis.get_ylim()[0], -0.1), max_error_without_nan)

                # absolute_axis.autoscale_view()

                ratio_axis.set_ylim(bottom=1)

                figure.tight_layout()
                plt.savefig(save_path / f"mri_flair_{metric}_{error}.png", dpi=600)
                # plt.show()


def plot_legend(block_shapes, colors, index_type_markers, index_types, save_path):
    plt.clf()
    plt.figure(figsize=(8, 0.8))
    legend = [Line2D([0], [0], marker="_", markersize=10, linestyle="", color="black", label=f"mean compression ratio")]
    for index_type, marker in zip(index_types, index_type_markers):
        legend.append(Line2D([0], [0], marker=marker, linestyle="", color="black", label=f"index type {index_type}"))
    for block_shape, color in zip(block_shapes, colors):
        legend.append(Patch(facecolor=color, label=f"{'×'.join(str(size) for size in block_shape)} blocks"))
    plt.legend(loc="center", ncol=len(legend) // 2, handles=legend)
    plt.gca().axis("off")
    # plt.savefig(save_path / f"mri_flair_legend.png")
    plt.savefig(save_path / f"mri_flair_legend.pdf")
    # plt.show()


if __name__ == "__main__":
    main()
