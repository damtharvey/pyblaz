import torch
from pyblaz.compression import PyBlaz


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
codec = PyBlaz(block_shape=(4, 4), dtype=torch.float32, index_dtype=torch.int8, device=device)

x = torch.randn(8, 8, device=device) * 2 + 3
compressed_x = codec.compress(x)
normalized_x = codec.decompress((compressed_x - compressed_x.mean()) / compressed_x.standard_deviation())

print(normalized_x.mean().item(), normalized_x.std(correction=0).item())
