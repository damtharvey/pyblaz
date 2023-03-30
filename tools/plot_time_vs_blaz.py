import matplotlib.pyplot as plt
import matplotlib.colors
import pandas as pd


def main():
    dataframe = pd.read_csv("results/time/results.csv")
    dataframe = dataframe[
        (dataframe["dimensions"] == 2)
        & (dataframe["dtype"] == "float64")
        & (dataframe["index_dtype"] == "int8")
        & (dataframe["block_size"] == 8)
    ]
    blaz_dataframe = pd.read_csv("results/time/blaz.csv")

    blaz_operation_names = {
        "compress": "compress",
        "decompress": "decompress",
        "add": "compressed_add",
        "multiply": "compressed_multiply",
    }
    colors = list(matplotlib.colors.TABLEAU_COLORS.keys())
    horizontal_values = [str(x) for x in sorted(dataframe["size"].unique())]

    for operation, color in zip(blaz_operation_names, colors):
        plt.plot(
            horizontal_values, dataframe[operation].astype(float) / 10**6, color=color, label=f"PyBlaz {operation}"
        )

    for (operation, blaz_operation), color in zip(blaz_operation_names.items(), colors):
        plt.plot(
            horizontal_values,
            blaz_dataframe[operation].astype(float) / 10**6,
            color=color,
            linestyle="dashed",
            label=f"Blaz {operation}",
        )

    plt.title("PyBlaz vs. Blaz Operation Time")
    plt.xlabel("array size")
    plt.ylabel("time (seconds)")
    plt.yscale("log")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig("results/time/plots/pyblaz_vs_blaz.pdf")
    # plt.show()


if __name__ == "__main__":
    main()
