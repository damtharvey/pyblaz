# PyBlaz

PyBlaz is an experimental tensor compression library that allows direct operations on compressed tensors without decompression. The library enables efficient computation with decreased memory requirements.

## Dependencies

Install PyTorch >=2.0.0. Go to https://pytorch.org/get-started and select your installation configuration from the table. Then copy the provided command and run it.

## Installation

```bash
pip install -e .
```

For development, install additional dependencies:

```bash
pip install -e ".[dev]"
```

## Example Usage

```python
import torch
from pyblaz.compression import PyBlaz

# Create a compressor with desired block shape and settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
codec = PyBlaz(
    block_shape=(4, 4),
    dtype=torch.float32,
    index_dtype=torch.int8,
    device=device
)

# Create sample data
x = torch.randn(8, 8, device=device) * 2 + 3

# Compress the tensor
compressed_x = codec.compress(x)

# Perform operations directly on compressed tensor
normalized_x = codec.decompress(
    (compressed_x - compressed_x.mean()) / compressed_x.standard_deviation()
)

print(f"Mean: {normalized_x.mean().item():.6f}, Std: {normalized_x.std(correction=0).item():.6f}")
```

## Compute Modes

PyBlaz supports different compute modes for optimized performance:

### Standard FP32 Mode
```python
codec = PyBlaz(
    block_shape=(16, 16),
    dtype=torch.float32,
    device=torch.device("cuda"),
    compute_mode="fp32"  # Default mode
)
```

### TensorFloat32 (TF32) Mode
On NVIDIA Ampere+ GPUs, you can use TF32 for faster matrix operations:
```python
codec = PyBlaz(
    block_shape=(16, 16),
    dtype=torch.float32,  # Required for TF32 mode
    device=torch.device("cuda"),
    compute_mode="tf32"  # Use TF32 for faster matrix operations
)
```

TF32 provides significant speedups with minimal precision loss. Note that TF32 mode requires `dtype=torch.float32`. To benchmark the performance:
```bash
python tests/benchmark_tf32.py --dimensions 2 --size 1024 --block-size 8
```

## Supported Operations

PyBlaz supports the following operations directly on compressed tensors:

- Basic arithmetic: addition, subtraction, multiplication by scalars, division by scalars
- Statistical operations: mean, variance, standard deviation, covariance
- Distance metrics: dot product, L2 norm, cosine similarity
- Advanced operations: structural similarity

## Running Tests

To run compression benchmarks:

```bash
python tests/test_compression.py --dimensions 2 --block-size 8 --max-size 256
```

To visualize the transforms:

```bash
python tests/test_transforms.py
```

## Funding Acknowledgement
This software was developed under the auspices of funding
under NSF 2217154, "Collaborative Research: PPoSS: Large: A comprehensive framework for efficient, scalable, and performance-portable tensor applications". 

## Citation
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
