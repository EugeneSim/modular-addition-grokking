"""
Define the neural network model for the modular addition task.
"""
import torch
from torch import nn


class ToyMLP(nn.Module):
    """
    Simple MLP for modular addition.
    Embeds inputs, concatenates them, and predicts the result.
    """
    def __init__(self, vocab_size=113, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(2 * embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        """
        Perform a forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 2) with integers in [0, vocab_size-1]
        
        Returns:
            Logits of shape (batch_size, vocab_size)
        """
        # Embed both inputs and concatenate them
        emb = self.embed(x)  # (batch_size, 2, embed_dim)
        emb = emb.view(x.size(0), -1)  # (batch_size, 2 * embed_dim)
        
        # Forward through the MLP layers
        out = self.fc1(emb)
        out = self.relu(out)
        out = self.fc2(out)
        
        return out
    
    def get_activations(self, x):
        """
        Get intermediate activations for interpretability analysis.
        
        Returns:
            Dictionary with 'embed', 'fc1_pre', 'fc1_post', 'fc2_pre'
        """
        activations = {}
        
        emb = self.embed(x)
        activations['embed'] = emb.view(x.size(0), -1)
        
        fc1_input = activations['embed']
        activations['fc1_pre'] = fc1_input
        
        fc1_output = self.fc1(fc1_input)
        activations['fc1_post'] = fc1_output
        
        relu_output = self.relu(fc1_output)
        activations['relu_post'] = relu_output
        
        fc2_output = self.fc2(relu_output)
        activations['fc2_pre'] = fc2_output
        
        return activations
