import csv

from compression import Compressor

import torch

import time
from tabulate import tabulate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rho_1 = []
with open("../data/OrzagTang_singleprecision/1.csv") as csv_file:
    next(csv_file)
    for row in csv.reader(csv_file, delimiter=","):
        rho_1.append(float(row[0]))

rho_2 = []
with open("../data/OrzagTang_singleprecision/2.csv") as csv_file:
    next(csv_file)
    for row in csv.reader(csv_file, delimiter=","):
        rho_2.append(float(row[0]))

a = torch.FloatTensor(rho_1[:251001]).reshape(501, 501).to(device)
b = torch.FloatTensor(rho_2[:251001]).reshape(501, 501).to(device)

dtype = torch.float64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
compressor = Compressor(block_shape=(4, 4), dtype=dtype, device=device)

compressed_a = compressor.compress(a)
compressed_b = compressor.compress(b)

# -----------------------------Time analysis for subtraction
begin1 = time.time()
decompressed_subtraction = compressor.decompress(compressed_a - compressed_b)
end1 = time.time()
print(f"Total runtime of the decompressed_subtraction is {end1 - begin1} secs")
subtraction = a - b

begin2 = time.time()
decompressed_a = compressor.decompress(compressed_a)
decompressed_b = compressor.decompress(compressed_b)
subtraction_compression_decompression = decompressed_a - decompressed_b
end2 = time.time()
print(f"Total runtime of the subtraction_compression_decompression is {end2 - begin2} secs")

# -----------------------------Accuracy analysis for subtraction
print(
    f"difference between actual subtraction and subtraction in compressed domain = {(subtraction - decompressed_subtraction).norm(torch.inf)} "
)
print(
    f"difference between actual subtraction and subtraction after decompressing the data = {(subtraction - subtraction_compression_decompression).norm(torch.inf)}"
)
time1 = end1 - begin1
accuracy1 = (subtraction - decompressed_subtraction).norm(torch.inf)
time2 = end2 - begin2
accuracy2 = (subtraction - subtraction_compression_decompression).norm(torch.inf)
# ---------------------------Time analysis for dot product(cosine difference)
begin3 = time.time()
decompressed_dotproduct = compressor.dot_product(compressed_a, compressed_b, 5, 6)
end3 = time.time()
dot_row_col = a[5] @ b[:, 6]

print(f"Total runtime of the dotproduct_compression_decompression is {end3 - begin3} secs")


begin4 = time.time()
decompressed_a = compressor.decompress(compressed_a)
decompressed_b = compressor.decompress(compressed_b)
dotproduct_compression_decompression = decompressed_a[5] @ decompressed_b[:, 6]
end4 = time.time()
dot_row_col = a[5] @ b[:, 6]

print(f"Total runtime of the dotproduct_compression_decompression is {end4 - begin4} secs")
time3 = end3 - begin3
accuracy3 = (dot_row_col - decompressed_dotproduct).norm(torch.inf)
time4 = end4 - begin4
accuracy4 = (dot_row_col - dotproduct_compression_decompression).norm(torch.inf)
# -----------------------------Accuracy analysis for dot product(cosine difference)
print(
    f"difference between actual dot product and dot product in compressed domain = {(dot_row_col - decompressed_dotproduct).norm(torch.inf)}"
)
print(
    f"difference between actual dot product and dot product after decompressing the data = {(dot_row_col - dotproduct_compression_decompression).norm(torch.inf)}"
)

data = [["Subtraction", time1, time2, accuracy1, accuracy2], ["Dot product", time3, time4, accuracy3, accuracy4]]

col_names = [
    "Operation",
    "Time in compressed domain ",
    "Time after decompression ",
    "Accuracy in compressed domain",
    "Accuracy after decompression",
]
print(tabulate(data, headers=col_names, tablefmt="fancy_grid"))
