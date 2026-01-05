"""
Train the model for the modular addition grokking experiment.
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from model import ToyMLP
from dataset import generate_modular_addition_dataset
import os


def train_model(p=113, embed_dim=128, hidden_dim=256, epochs=1000, 
                lr=1e-3, weight_decay=1e-2, train_size=10000, test_size=2000,
                eval_interval=50, save_interval=500, checkpoint_dir='checkpoints'):
    """
    Train the model and observe the grokking phenomenon.
    
    Returns:
        model, train_losses, test_accs (list of (epoch, accuracy) tuples)
    """
    # Create the checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Generate the dataset
    print("Generating dataset...")
    train_loader, test_loader, p = generate_modular_addition_dataset(
        p=p, train_size=train_size, test_size=test_size
    )
    
    # Initialize the model
    model = ToyMLP(vocab_size=p, embed_dim=embed_dim, hidden_dim=hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Track training progress
    train_losses = []
    test_accs = []
    
    print(f"Starting training for {epochs} epochs...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for inputs, targets in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Evaluate test accuracy periodically
        if epoch % eval_interval == 0:
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in test_loader:
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            test_acc = correct / total
            test_accs.append((epoch, test_acc))
            
            print(f"Epoch {epoch:5d}: Train Loss {avg_loss:.4f}, Test Acc {test_acc:.4f}")
            
            # Check for grokking (sudden accuracy jump)
            if len(test_accs) >= 2:
                prev_acc = test_accs[-2][1]
                if test_acc - prev_acc > 0.3:  # Sudden jump of >30%
                    print(f"  ⚡ Potential grokking detected! Accuracy jumped from {prev_acc:.4f} to {test_acc:.4f}")
        
        # Save checkpoints periodically
        if epoch % save_interval == 0 and epoch > 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'test_accs': test_accs,
            }, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")
    
    # Save the final model
    final_path = os.path.join(checkpoint_dir, 'model_final.pth')
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'test_accs': test_accs,
    }, final_path)
    print(f"\nTraining complete! Final model saved: {final_path}")
    
    return model, train_losses, test_accs


def plot_training_curves(train_losses, test_accs, save_path='training_curves.png'):
    """
    Plot training loss and test accuracy curves.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot training loss
    ax1.plot(train_losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Over Time')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Plot test accuracy
    epochs = [e for e, _ in test_accs]
    accs = [a for _, a in test_accs]
    ax2.plot(epochs, accs, marker='o', markersize=3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title('Test Accuracy Over Time (Grokking Detection)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    # Highlight the grokking point if detected
    if len(accs) >= 2:
        for i in range(1, len(accs)):
            if accs[i] - accs[i-1] > 0.3:
                ax2.axvline(x=epochs[i], color='r', linestyle='--', alpha=0.5, label='Grokking')
                ax2.legend()
                break
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Training curves saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    # Train the model
    model, train_losses, test_accs = train_model(
        p=113,
        epochs=1000,
        eval_interval=50,
        save_interval=500
    )
    
    # Plot the results
    plot_training_curves(train_losses, test_accs)
    
    print("\nTraining summary:")
    print(f"Final test accuracy: {test_accs[-1][1]:.4f}")
    print(f"Final training loss: {train_losses[-1]:.4f}")
