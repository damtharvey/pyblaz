import pandas as pd


def main():
    df = pd.read_csv("../results/mri/mri_metrics.csv")
    as_tuples = tuple(tuple(int(x) for x in shape.split("×")) for shape in df.original_shape)
    print(f"Shortest: {min(as_tuples, key=lambda x: x[0])}")
    print(f"Longest: {max(as_tuples, key=lambda x: x[0])}")
    print(f"Mean: {sum(t[0] for t in as_tuples) / len(as_tuples)}")


if __name__ == "__main__":
    main()
