import pathlib

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors


def main():
    with open("results/ShallowWaters/weighted_dct_shallow_water.csv") as file:
        dataframe = pd.read_csv(file)

    figure_path = pathlib.Path("results/ShallowWaters/figures")
    figure_path.mkdir(parents=True, exist_ok=True)

    colors = list(matplotlib.colors.TABLEAU_COLORS.keys())

    plt.plot(range(500), dataframe["fastmath_vs_o3"], color=colors[0], label="FC1 vs FC0")
    plt.plot(range(500), dataframe["ftz_vs_o3"], color=colors[2], label="FC2 vs FC0")
    plt.title("Absolute difference between sums of weighted absolute coefficients", wrap=True, loc="center")
    plt.ylabel("absolute difference")
    plt.xlabel("time-step")
    plt.legend()
    plt.savefig("results/ShallowWaters/figures/weighted_dct_difference.pdf")

    plt.clf()
    horizontal_values = [str(x) for x in range(20)]
    plt.plot(horizontal_values, dataframe["fastmath_vs_o3"].head(20), color=colors[0], label="FC1 vs FC0")
    plt.plot(horizontal_values, dataframe["ftz_vs_o3"].head(20), color=colors[2], label="FC2 vs FC0")
    plt.title("(Zoomed) Absolute difference between sums of weighted absolute coefficients", wrap=True, loc="center")
    plt.ylabel("absolute difference")
    plt.xlabel("time-step")
    plt.legend()
    plt.savefig("results/ShallowWaters/figures/weighted_dct_difference_xlim_0_19.pdf")


if __name__ == "__main__":
    main()
