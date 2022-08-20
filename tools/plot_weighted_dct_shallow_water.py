import pandas as pd
import matplotlib.pyplot as plt


def main():
    with open("results/ShallowWaters/weighted_dct_shallow_water.csv") as file:
        dataframe = pd.read_csv(file)

    plt.plot(range(500), dataframe["fastmath_vs_o3"])
    plt.plot(range(500), dataframe["ftz_vs_o3"])
    plt.show()


if __name__ == "__main__":
    main()
