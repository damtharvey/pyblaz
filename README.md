# Dependencies

Install PyTorch >=2.0. Go to https://pytorch.org/get-started and select your installation configuration from the table. Then copy the provided command and run it.

# Installation

```bash
pip install -e .
```

# Example Usage

```python
import torch
from pyblaz.compression import PyBlaz


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
codec = PyBlaz(block_shape=(4, 4), dtype=torch.float32, index_dtype=torch.int8, device=device)

x = torch.randn(8, 8, device=device) * 2 + 3
compressed_x = codec.compress(x)
normalized_x = codec.decompress((compressed_x - compressed_x.mean()) / compressed_x.standard_deviation())

print(normalized_x.mean().item(), normalized_x.std(correction=0).item())
```

# Funding Acknowledgement
This software was developed under the auspices of funding
under NSF 2217154, "Collaborative Research: PPoSS: Large: A comprehensive framework for efficient, scalable, and performance-portable tensor applications". 

# Citation
```
@inproceedings{10.1145/3624062.3625122,
author = {Agarwal, Tripti and Dam, Harvey and Sadayappan, Ponnuswamy and Gopalakrishnan, Ganesh and Khalifa, Dorra Ben and Martel, Matthieu},
title = {What Operations can be Performed Directly on Compressed Arrays, and with What Error?},
year = {2023},
isbn = {9798400707858},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3624062.3625122},
doi = {10.1145/3624062.3625122},
booktitle = {Proceedings of the SC '23 Workshops of The International Conference on High Performance Computing, Network, Storage, and Analysis},
pages = {254–262},
numpages = {9},
keywords = {arrays, data compression, floating-point arithmetic, high-performance computing, parallel computing, tensors},
location = {Denver, CO, USA},
series = {SC-W '23}
}
```
