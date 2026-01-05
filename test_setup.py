"""
Quick test to verify the setup is working correctly.
"""
import torch
import sys

def test_imports():
    """Test that all required packages are installed."""
    print("Testing imports...")
    try:
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        print("✓ Core packages imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def test_dataset():
    """Test dataset generation."""
    print("\nTesting dataset generation...")
    try:
        from dataset import generate_modular_addition_dataset
        train_loader, test_loader, p = generate_modular_addition_dataset(
            p=13, train_size=100, test_size=20, batch_size=16
        )
        print(f"✓ Dataset generated: p={p}, train batches={len(train_loader)}, test batches={len(test_loader)}")
        return True
    except Exception as e:
        print(f"✗ Dataset error: {e}")
        return False

def test_model():
    """Test model creation and forward pass."""
    print("\nTesting model...")
    try:
        from model import ToyMLP
        model = ToyMLP(vocab_size=13, embed_dim=16, hidden_dim=32)
        x = torch.tensor([[5, 7], [2, 3]])
        output = model(x)
        print(f"✓ Model created: input shape {x.shape}, output shape {output.shape}")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        return True
    except Exception as e:
        print(f"✗ Model error: {e}")
        return False

def test_training_step():
    """Test a single training step."""
    print("\nTesting training step...")
    try:
        from model import ToyMLP
        from dataset import generate_modular_addition_dataset
        import torch.nn as nn
        
        model = ToyMLP(vocab_size=13, embed_dim=16, hidden_dim=32)
        train_loader, _, _ = generate_modular_addition_dataset(p=13, train_size=50, batch_size=8)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        # One step
        inputs, targets = next(iter(train_loader))
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"✓ Training step successful: loss={loss.item():.4f}")
        return True
    except Exception as e:
        print(f"✗ Training error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("SETUP VERIFICATION TEST")
    print("=" * 60)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Dataset", test_dataset()))
    results.append(("Model", test_model()))
    results.append(("Training", test_training_step()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = all(result for _, result in results)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:15s}: {status}")
    
    if all_passed:
        print("\n✓ All tests passed! Setup is ready.")
        print("\nNext steps:")
        print("  1. Quick test: python example_quick_start.py")
        print("  2. Full training: python train.py")
        print("  3. Analysis: python interpretability.py (after training)")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")
        sys.exit(1)
