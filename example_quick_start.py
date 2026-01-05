"""
Quick start example: train a small model for demonstration.
Use this for faster iteration before running full training.
"""
import torch
from model import ToyMLP
from dataset import generate_modular_addition_dataset
from train import train_model, plot_training_curves

if __name__ == "__main__":
    print("Quick Start: Training a small model for demonstration")
    print("=" * 60)
    
    # Use smaller parameters for a quick demo
    model, train_losses, test_accs = train_model(
        p=97,  # Smaller prime for faster training
        embed_dim=64,  # Smaller embedding
        hidden_dim=128,  # Smaller hidden layer
        epochs=5000,  # Fewer epochs for demo
        lr=1e-3,
        weight_decay=1e-2,
        train_size=5000,
        test_size=1000,
        eval_interval=200,  # Evaluate less frequently
        save_interval=2000
    )
    
    # Plot the results
    plot_training_curves(train_losses, test_accs, save_path='results/quick_start_curves.png')
    
    print("\nQuick start complete!")
    print("For full training, run: python train.py")
