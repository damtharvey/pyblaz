import torch
import numpy as np
import matplotlib.pyplot as plt
from pyblaz.transforms import cosine, haar


def test_cosine_transform():
    """
    Test the cosine transform function.
    """
    block_size = 8
    # Create transform matrix
    transform_matrix = torch.tensor(
        [[cosine(block_size, element, frequency, False) for frequency in range(block_size)] 
         for element in range(block_size)]
    )
    
    # Test orthogonality property
    identity = transform_matrix @ transform_matrix.T
    print("Orthogonality check (should be close to identity):")
    print(f"Max deviation from identity: {(identity - torch.eye(block_size)).abs().max().item()}")
    
    # Plot basis functions
    plt.figure(figsize=(12, 8))
    for frequency in range(min(8, block_size)):
        plt.plot(transform_matrix[:, frequency].numpy(), 
                 label=f"Frequency {frequency}")
    plt.title("Cosine Transform Basis Functions")
    plt.legend()
    plt.grid(True)
    plt.savefig("cosine_basis.png")
    plt.close()


def test_haar_transform():
    """
    Test the Haar transform function.
    """
    block_size = 8
    # Create transform matrix
    transform_matrix = torch.tensor(
        [[haar(block_size, element, frequency, False) for frequency in range(block_size)] 
         for element in range(block_size)]
    )
    
    # Test orthogonality property
    identity = transform_matrix @ transform_matrix.T
    print("Orthogonality check (should be close to identity):")
    print(f"Max deviation from identity: {(identity - torch.eye(block_size)).abs().max().item()}")
    
    # Plot basis functions
    plt.figure(figsize=(12, 8))
    for frequency in range(min(8, block_size)):
        plt.plot(transform_matrix[:, frequency].numpy(), 
                 label=f"Order {frequency}")
    plt.title("Haar Transform Basis Functions")
    plt.legend()
    plt.grid(True)
    plt.savefig("haar_basis.png")
    plt.close()


if __name__ == "__main__":
    test_cosine_transform()
    test_haar_transform() 