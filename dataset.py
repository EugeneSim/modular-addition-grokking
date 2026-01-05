"""
Generate datasets for the modular addition task.
Create training and test splits for (a + b) mod p.
"""
import torch
from torch.utils.data import DataLoader, TensorDataset


def generate_modular_addition_dataset(p=113, train_size=10000, test_size=2000, split_ratio=0.8, batch_size=64):
    """
    Generate a dataset for the modular addition task.
    
    Args:
        p: Modulo prime (default: 113)
        train_size: Maximum training samples to create
        test_size: Maximum test samples to create
        split_ratio: Fraction of data to use for training
        batch_size: Batch size for DataLoaders
    
    Returns:
        train_loader, test_loader, p
    """
    # Generate all possible pairs (a, b) in [0, p-1]
    all_inputs = torch.cartesian_prod(torch.arange(p), torch.arange(p))
    all_labels = (all_inputs[:, 0] + all_inputs[:, 1]) % p
    
    # Shuffle and split the data
    perm = torch.randperm(p * p)
    total_samples = p * p
    train_end_idx = int(split_ratio * total_samples)
    
    train_idx = perm[:train_end_idx]
    test_idx = perm[train_end_idx:]
    
    # Limit to the requested sizes
    train_inputs = all_inputs[train_idx][:train_size]
    train_labels = all_labels[train_idx][:train_size]
    test_inputs = all_inputs[test_idx][:test_size]
    test_labels = all_labels[test_idx][:test_size]
    
    # Create DataLoaders for efficient batch processing
    train_dataset = TensorDataset(train_inputs, train_labels)
    test_dataset = TensorDataset(test_inputs, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, p


if __name__ == "__main__":
    # Test dataset generation to verify everything works
    train_loader, test_loader, p = generate_modular_addition_dataset()
    print(f"Dataset generated with p={p}")
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Check a sample to verify correctness
    sample_inputs, sample_labels = next(iter(train_loader))
    print(f"Sample batch shape: {sample_inputs.shape}")
    print(f"Sample: {sample_inputs[0]} -> {sample_labels[0]} (should be {(sample_inputs[0][0] + sample_inputs[0][1]) % p})")
