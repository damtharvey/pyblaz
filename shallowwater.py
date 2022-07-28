from compression import Compressor

import torch

txt_file0 = open("./ShallowWatersEquations/output/0.txt", "r")
file_content0 = txt_file0.read()


content_list0 = file_content0.split("\n")

x = 80
list_of_lists0 = [content_list0[i:i+x] for i in range(0, len(content_list0), x)]
newlist0 = []
for word in list_of_lists0[0]:
    word = word.split(",")
    newlist0.append(word)  

newlist0 = newlist0[:-1]
final_list0 = []
for i in newlist0:
    temp = i[0].split(" ")
    temp1 = []
    for j in temp[:-1]:
        temp1.append(float(j))
    final_list0.append(temp1)
print(final_list0[0])


txt_file1 = open("./ShallowWatersEquations/output/1.txt", "r")
file_content1 = txt_file1.read()


content_list1 = file_content1.split("\n")

x = 80
list_of_lists1 = [content_list1[i:i+x] for i in range(0, len(content_list1), x)]
newlist1 = []
for word in list_of_lists1[0]:
    word = word.split(",")
    newlist1.append(word)  

newlist1 = newlist1[:-1]
final_list1 = []
for i in newlist1:
    temp = i[0].split(" ")
    temp1 = []
    for j in temp[:-1]:
        temp1.append(float(j))
    final_list1.append(temp1)
print(len(final_list1))

dtype = torch.float64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

compressor = Compressor(block_shape=(4, 4), dtype=dtype, device=device)

a = torch.FloatTensor(final_list0)
b = torch.FloatTensor(final_list1)

compressed_a = compressor.compress(a)
compressed_b = compressor.compress(b)

decompressed_subtraction = compressor.decompress(compressed_a - compressed_b)
print(sum(sum(decompressed_subtraction)))