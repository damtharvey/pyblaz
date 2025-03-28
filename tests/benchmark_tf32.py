import torch
import time
import numpy as np
import argparse
from pyblaz.compression import PyBlaz


def benchmark_tf32_performance():
    """
    Benchmark the performance improvement of TF32 over regular FP32.
    """
    parser = argparse.ArgumentParser(description="Benchmark TF32 vs FP32 performance in PyBlaz")
    parser.add_argument("--dimensions", type=int, default=2, help="Number of dimensions in test tensors")
    parser.add_argument("--size", type=int, default=1024, help="Size of the tensor in each dimension")
    parser.add_argument("--block-size", type=int, default=8, help="Size of compression blocks")
    parser.add_argument("--repeats", type=int, default=10, help="Number of repetitions for each test")
    args = parser.parse_args()

    # Only proceed if CUDA is available (TF32 is CUDA-specific)
    if not torch.cuda.is_available():
        print("CUDA is not available. TF32 requires NVIDIA Ampere or newer GPUs.")
        return

    device = torch.device("cuda")
    block_shape = (args.block_size,) * args.dimensions
    tensor_shape = (args.size,) * args.dimensions

    # Create random test data
    print(f"Creating {args.dimensions}D tensor of shape {tensor_shape}...")
    x = torch.randn(tensor_shape, dtype=torch.float32, device=device)

    # Create compressors with and without TF32
    compressor_fp32 = PyBlaz(
        block_shape=block_shape, dtype=torch.float32, index_dtype=torch.int16, device=device, compute_mode="fp32"
    )

    compressor_tf32 = PyBlaz(
        block_shape=block_shape, dtype=torch.float32, index_dtype=torch.int16, device=device, compute_mode="tf32"
    )

    # Warmup
    print("Warming up...")
    _ = compressor_fp32.compress(x)
    _ = compressor_tf32.compress(x)
    torch.cuda.synchronize()

    # Benchmark compression
    print(f"Running {args.repeats} iterations for each mode...")

    # FP32 timings
    fp32_times = []
    for i in range(args.repeats):
        start = time.time()
        compressed = compressor_fp32.compress(x)
        decompressed = compressor_fp32.decompress(compressed)
        torch.cuda.synchronize()
        fp32_times.append(time.time() - start)

    # TF32 timings
    tf32_times = []
    for i in range(args.repeats):
        start = time.time()
        compressed = compressor_tf32.compress(x)
        decompressed = compressor_tf32.decompress(compressed)
        torch.cuda.synchronize()
        tf32_times.append(time.time() - start)

    # Calculate statistics
    fp32_mean = np.mean(fp32_times)
    fp32_std = np.std(fp32_times)
    tf32_mean = np.mean(tf32_times)
    tf32_std = np.std(tf32_times)
    speed_ratio = fp32_mean / tf32_mean if tf32_mean > 0 else 0

    # Print results
    print("\nResults:")
    print("=" * 60)
    print(f"{'Mode':<10} | {'Mean Time (s)':<15} | {'Std Dev (s)':<15} | {'Speed':<10}")
    print("-" * 60)
    print(f"{'FP32':<10} | {fp32_mean:<15.5f} | {fp32_std:<15.5f} | {'baseline':<10}")
    print(f"{'TF32':<10} | {tf32_mean:<15.5f} | {tf32_std:<15.5f} | {speed_ratio:.2f}x as fast")
    print("=" * 60)
    print(f"\nTF32 is {speed_ratio:.2f} times as fast as FP32")

    # Check for numerical differences
    print("\nNumerical Accuracy Test:")
    # Compress and decompress the same tensor with both modes
    x_sample = torch.randn((64,) * args.dimensions, dtype=torch.float32, device=device)

    compressed_fp32 = compressor_fp32.compress(x_sample)
    decompressed_fp32 = compressor_fp32.decompress(compressed_fp32)

    compressed_tf32 = compressor_tf32.compress(x_sample)
    decompressed_tf32 = compressor_tf32.decompress(compressed_tf32)

    # Compute differences
    abs_diff = (decompressed_fp32 - decompressed_tf32).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()
    rel_error = abs_diff.sum() / decompressed_fp32.abs().sum()

    print(f"Max absolute difference: {max_diff:.8e}")
    print(f"Mean absolute difference: {mean_diff:.8e}")
    print(f"Relative error: {rel_error:.8e}")


if __name__ == "__main__":
    benchmark_tf32_performance()
