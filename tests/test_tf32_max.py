import torch
from pyblaz.compression import PyBlaz


class TestPyBlaz(PyBlaz):
    """Subclass of PyBlaz that exposes the intermediate transformed tensor."""

    def get_transformed_blocks(self, tensor):
        """Access the tensor after blockwise transform but before binning."""
        # Block the tensor as in the compression pipeline
        blocked = self.compressor.block(tensor.to(self.dtype).to(self.device))

        # Apply the blockwise transform
        transformed = self.blockwise_transform(blocked)

        return transformed


def test_tf32_max_value():
    """Test with the maximum TF32 representable value."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a codec instance with TF32 enabled
    codec = TestPyBlaz(
        block_shape=(4, 4), dtype=torch.float32, device=device, compute_mode="tf32"  # Specifically use TF32 mode
    )

    tf32_max = (2 - 2**-10) * (1 << 127)

    # Create tensor with max TF32 value
    x = torch.ones(4, 4, device=device) * tf32_max

    print(f"Testing with TF32 max value: {tf32_max}")
    print(f"Input tensor contains NaN: {torch.isnan(x).any().item()}")
    print(f"Input tensor contains Inf: {torch.isinf(x).any().item()}")

    # Apply blockwise transform
    transformed = codec.get_transformed_blocks(x)

    # Check for NaN/Inf in the result
    has_nan = torch.isnan(transformed).any().item()
    has_inf = torch.isinf(transformed).any().item()

    print(f"After transform - Contains NaN: {has_nan}")
    print(f"After transform - Contains Inf: {has_inf}")

    if has_nan or has_inf:
        print("The blockwise transform produced NaN or Inf with TF32 max value")
        print(f"Number of NaN values: {torch.isnan(transformed).sum().item()}")
        print(f"Number of Inf values: {torch.isinf(transformed).sum().item()}")

        # Find where these values are
        nan_positions = torch.nonzero(torch.isnan(transformed))
        inf_positions = torch.nonzero(torch.isinf(transformed))

        if len(nan_positions) > 0:
            print(f"NaN positions: {nan_positions[:5]}...")
        if len(inf_positions) > 0:
            print(f"Inf positions: {inf_positions[:5]}...")

        # Print some sample values
        print("\nSample of transformed tensor values:")
        print(transformed.flatten()[:16])
    else:
        print("No NaN or Inf values were produced.")

    # Try with a slightly smaller value (99% of max)
    print("\n\nTesting with 99% of TF32 max value")
    x_smaller = torch.ones(4, 4, device=device) * (tf32_max * 0.99)

    # Apply blockwise transform
    transformed_smaller = codec.get_transformed_blocks(x_smaller)

    # Check for NaN/Inf in the result
    has_nan_smaller = torch.isnan(transformed_smaller).any().item()
    has_inf_smaller = torch.isinf(transformed_smaller).any().item()

    print(f"After transform - Contains NaN: {has_nan_smaller}")
    print(f"After transform - Contains Inf: {has_inf_smaller}")

    return transformed


if __name__ == "__main__":
    test_tf32_max_value()
