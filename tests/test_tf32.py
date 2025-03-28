import torch
import pytest
from pyblaz.compression import PyBlaz


def test_tf32_basic_functionality():
    """
    Test that TF32 can be enabled and used without errors.
    Skips test if CUDA is not available.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping TF32 test")

    # Check if running on Ampere or newer GPU (required for TF32)
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)

    # Log the GPU details
    print(f"Testing on GPU: {gpu_name}")

    # Create a small tensor
    x = torch.randn(32, 32, device=device)

    # Test with TF32 enabled
    compressor_tf32 = PyBlaz(
        block_shape=(8, 8), dtype=torch.float32, index_dtype=torch.int16, device=device, compute_mode="tf32"
    )

    # Test without TF32
    compressor_fp32 = PyBlaz(
        block_shape=(8, 8), dtype=torch.float32, index_dtype=torch.int16, device=device, compute_mode="fp32"
    )

    # Compress and decompress with both
    compressed_tf32 = compressor_tf32.compress(x)
    decompressed_tf32 = compressor_tf32.decompress(compressed_tf32)

    compressed_fp32 = compressor_fp32.compress(x)
    decompressed_fp32 = compressor_fp32.decompress(compressed_fp32)

    # Check that both versions work and produce similar results
    abs_diff = (decompressed_tf32 - decompressed_fp32).abs()
    mean_diff = abs_diff.mean().item()
    max_diff = abs_diff.max().item()

    print(f"Mean difference between TF32 and FP32: {mean_diff:.8e}")
    print(f"Max difference between TF32 and FP32: {max_diff:.8e}")

    # We expect small differences due to TF32's reduced precision
    assert abs_diff.mean().item() < 1e-3, "Mean difference too large"

    # Verify that original tensor is recovered with reasonable accuracy in both cases
    tf32_error = (x - decompressed_tf32).abs().mean().item()
    fp32_error = (x - decompressed_fp32).abs().mean().item()

    print(f"TF32 reconstruction error: {tf32_error:.8e}")
    print(f"FP32 reconstruction error: {fp32_error:.8e}")

    assert tf32_error < 1e-2, "TF32 reconstruction error too large"
    assert fp32_error < 1e-2, "FP32 reconstruction error too large"


def test_tf32_requires_float32():
    """
    Test that TF32 mode can only be used with torch.float32 dtype.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping TF32 test")

    device = torch.device("cuda")

    # This should work
    PyBlaz(block_shape=(8, 8), dtype=torch.float32, index_dtype=torch.int16, device=device, compute_mode="tf32")

    # This should raise an error
    with pytest.raises(ValueError, match="TF32 compute mode can only be used with torch.float32 dtype"):
        PyBlaz(
            block_shape=(8, 8),
            dtype=torch.float16,  # Wrong dtype
            index_dtype=torch.int16,
            device=device,
            compute_mode="tf32",
        )


if __name__ == "__main__":
    test_tf32_basic_functionality()
