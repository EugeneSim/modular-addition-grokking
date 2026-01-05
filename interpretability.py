"""
Perform mechanistic interpretability analysis for the modular addition model.
Analyze internal activations, identify important neurons, and perform causal interventions.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from model import ToyMLP
from dataset import generate_modular_addition_dataset
import os


def load_model(checkpoint_path, p=113, embed_dim=128, hidden_dim=256):
    """Load a trained model from a checkpoint."""
    model = ToyMLP(vocab_size=p, embed_dim=embed_dim, hidden_dim=hidden_dim)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint


def analyze_activations(model, p=113, num_samples=1000):
    """
    Analyze model activations across different inputs.
    
    Returns:
        Dictionary with activation statistics
    """
    print("Analyzing activations...")
    
    # Generate sample inputs
    all_inputs = torch.cartesian_prod(torch.arange(p), torch.arange(p))
    sample_indices = torch.randperm(p * p)[:num_samples]
    sample_inputs = all_inputs[sample_indices]
    
    activations_dict = {
        'embed': [],
        'fc1_pre': [],
        'fc1_post': [],
        'relu_post': [],
        'fc2_pre': []
    }
    
    with torch.no_grad():
        for i in range(0, len(sample_inputs), 64):
            batch = sample_inputs[i:i+64]
            acts = model.get_activations(batch)
            
            for key in activations_dict:
                activations_dict[key].append(acts[key].cpu())
    
    # Concatenate all batches
    for key in activations_dict:
        activations_dict[key] = torch.cat(activations_dict[key], dim=0)
    
    # Compute statistics
    stats = {}
    for key, acts in activations_dict.items():
        stats[key] = {
            'mean': acts.mean(dim=0),
            'std': acts.std(dim=0),
            'max': acts.max(dim=0)[0],
            'min': acts.min(dim=0)[0],
            'abs_mean': acts.abs().mean(dim=0)
        }
    
    return activations_dict, stats


def identify_important_neurons(model, p=113, top_k=20):
    """
    Identify neurons that are most active/important for the task.
    """
    print("Identifying important neurons...")
    
    activations_dict, stats = analyze_activations(model, p, num_samples=2000)
    
    # Focus on relu_post (after first layer) - these are the "neurons"
    relu_acts = activations_dict['relu_post']  # (num_samples, hidden_dim)
    
    # Find neurons with highest average absolute activation
    neuron_importance = relu_acts.abs().mean(dim=0)
    top_neurons = neuron_importance.topk(top_k)
    
    print(f"\nTop {top_k} most active neurons:")
    for i, (neuron_idx, importance) in enumerate(zip(top_neurons.indices, top_neurons.values)):
        print(f"  Neuron {neuron_idx.item():3d}: importance = {importance.item():.4f}")
    
    return top_neurons.indices, neuron_importance


def plot_activation_heatmap(model, p=113, layer='relu_post', save_path='activation_heatmap.png'):
    """
    Plot a heatmap of activations for different input pairs.
    """
    print(f"Generating activation heatmap for {layer}...")
    # Ensure the directory exists
    dir_path = os.path.dirname(save_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    # Sample a subset for visualization (full p*p is too large)
    sample_size = min(500, p * p)
    all_inputs = torch.cartesian_prod(torch.arange(p), torch.arange(p))
    sample_indices = torch.randperm(p * p)[:sample_size]
    sample_inputs = all_inputs[sample_indices]
    
    activations_list = []
    sums_list = []
    
    with torch.no_grad():
        for i in range(0, len(sample_inputs), 64):
            batch = sample_inputs[i:i+64]
            acts = model.get_activations(batch)
            layer_acts = acts[layer]
            
            # Average over batch dimension for visualization
            activations_list.append(layer_acts.mean(dim=0).cpu())
            sums_list.append((batch[:, 0] + batch[:, 1]).cpu())
    
    # Get the top neurons
    all_acts = torch.stack(activations_list)
    top_neurons, _ = identify_important_neurons(model, p, top_k=10)
    
    # Plot the heatmap for top neurons
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a matrix: rows = samples, cols = top neurons
    heatmap_data = all_acts[:, top_neurons[:10]].numpy()
    
    im = ax.imshow(heatmap_data.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Neuron Index (Top 10)')
    ax.set_title(f'Activation Heatmap: {layer} (Top 10 Neurons)')
    ax.set_yticks(range(len(top_neurons[:10])))
    ax.set_yticklabels([f'N{idx.item()}' for idx in top_neurons[:10]])
    plt.colorbar(im, ax=ax, label='Activation Value')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Activation heatmap saved to {save_path}")
    plt.close()


def causal_intervention(model, test_loader, neuron_idx, p=113):
    """
    Perform a causal intervention: ablate a neuron and measure its impact on accuracy.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        neuron_idx: Index of neuron to ablate
    """
    print(f"\nPerforming causal intervention: ablating neuron {neuron_idx}...")
    
    # Measure the original model accuracy
    model.eval()
    correct_original = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct_original += predicted.eq(targets).sum().item()
    
    acc_original = correct_original / total
    
    # Create a patched model (zero out specific neuron)
    def patch_forward(self, x):
        """Patched forward that zeros out a specific neuron."""
        emb = self.embed(x)
        emb = emb.view(x.size(0), -1)
        out = self.fc1(emb)
        out = self.relu(out)
        
        # Zero out the specified neuron
        out[:, neuron_idx] = 0
        
        out = self.fc2(out)
        return out
    
    # Temporarily replace the forward method
    original_forward = model.forward
    model.forward = lambda x: patch_forward(model, x)
    
    # Test the patched model
    correct_patched = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct_patched += predicted.eq(targets).sum().item()
    
    acc_patched = correct_patched / total
    
    # Restore the original forward method
    model.forward = original_forward
    
    impact = acc_original - acc_patched
    print(f"Original accuracy: {acc_original:.4f}")
    print(f"Patched accuracy: {acc_patched:.4f}")
    print(f"Impact: {impact:.4f} ({impact/acc_original*100:.2f}% relative drop)")
    
    return acc_original, acc_patched, impact


def analyze_fourier_components(model, p=113, num_samples=1000):
    """
    Test the hypothesis that the model might use Fourier-like embeddings for modular arithmetic.
    Analyze if activations correlate with sine/cosine patterns.
    """
    print("\nAnalyzing potential Fourier components...")
    
    # Generate samples
    all_inputs = torch.cartesian_prod(torch.arange(p), torch.arange(p))
    sample_indices = torch.randperm(p * p)[:num_samples]
    sample_inputs = all_inputs[sample_indices]
    
    # Compute sums
    sums = (sample_inputs[:, 0] + sample_inputs[:, 1]) % p
    
    # Get activations
    activations_dict, _ = analyze_activations(model, p, num_samples=num_samples)
    relu_acts = activations_dict['relu_post']  # (num_samples, hidden_dim)
    
    # Test correlation with sine/cosine of sum
    correlations = []
    for freq in range(1, min(20, p//2)):
        sine_pattern = torch.sin(2 * np.pi * freq * sums.float() / p)
        cosine_pattern = torch.cos(2 * np.pi * freq * sums.float() / p)
        
        for neuron_idx in range(relu_acts.shape[1]):
            neuron_acts = relu_acts[:, neuron_idx]
            
            # Compute correlation
            sine_corr = torch.corrcoef(torch.stack([neuron_acts, sine_pattern]))[0, 1].item()
            cosine_corr = torch.corrcoef(torch.stack([neuron_acts, cosine_pattern]))[0, 1].item()
            
            max_corr = max(abs(sine_corr), abs(cosine_corr))
            if max_corr > 0.3:  # Significant correlation
                correlations.append({
                    'neuron': neuron_idx,
                    'freq': freq,
                    'sine_corr': sine_corr,
                    'cosine_corr': cosine_corr,
                    'max_corr': max_corr
                })
    
    if correlations:
        print(f"Found {len(correlations)} neurons with significant Fourier-like patterns:")
        for corr in sorted(correlations, key=lambda x: x['max_corr'], reverse=True)[:10]:
            print(f"  Neuron {corr['neuron']:3d}, freq={corr['freq']:2d}: "
                  f"max_corr={corr['max_corr']:.3f}")
    else:
        print("No strong Fourier patterns detected (threshold: 0.3)")
    
    return correlations


def run_full_analysis(checkpoint_path='checkpoints/model_final.pth', p=113, 
                      embed_dim=128, hidden_dim=256):
    """
    Run the complete interpretability analysis suite.
    """
    print("=" * 60)
    print("MECHANISTIC INTERPRETABILITY ANALYSIS")
    print("=" * 60)
    
    # Load the model
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using train.py")
        return
    
    model, checkpoint = load_model(checkpoint_path, p, embed_dim, hidden_dim)
    print(f"Model loaded from {checkpoint_path}")
    
    # Generate the test loader
    _, test_loader, _ = generate_modular_addition_dataset(p=p, test_size=2000)
    
    # 1. Identify important neurons
    top_neurons, neuron_importance = identify_important_neurons(model, p, top_k=20)
    
    # 2. Plot the activation heatmap
    plot_activation_heatmap(model, p, layer='relu_post', 
                           save_path='results/activation_heatmap.png')
    
    # 3. Perform causal interventions on top neurons
    print("\n" + "=" * 60)
    print("CAUSAL INTERVENTIONS")
    print("=" * 60)
    intervention_results = []
    for neuron_idx in top_neurons[:5]:  # Test top 5
        acc_orig, acc_patched, impact = causal_intervention(model, test_loader, 
                                                           neuron_idx.item(), p)
        intervention_results.append({
            'neuron': neuron_idx.item(),
            'acc_original': acc_orig,
            'acc_patched': acc_patched,
            'impact': impact
        })
    
    # 4. Perform Fourier analysis
    print("\n" + "=" * 60)
    print("FOURIER COMPONENT ANALYSIS")
    print("=" * 60)
    fourier_results = analyze_fourier_components(model, p)
    
    # Save the results
    os.makedirs('results', exist_ok=True)
    results_summary = {
        'top_neurons': top_neurons.tolist(),
        'neuron_importance': neuron_importance.tolist(),
        'interventions': intervention_results,
        'fourier_components': fourier_results
    }
    
    import json
    with open('results/analysis_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Analysis complete! Results saved to results/")
    print("=" * 60)


if __name__ == "__main__":
    run_full_analysis()
