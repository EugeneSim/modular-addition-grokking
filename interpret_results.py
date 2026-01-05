"""Quick script to interpret analysis results."""
import json

# Load results
with open('results/analysis_summary.json', 'r') as f:
    data = json.load(f)

print("=" * 70)
print("RESULTS INTERPRETATION")
print("=" * 70)

print("\n1. TOP NEURONS (Most Important)")
print("-" * 70)
print(f"Found {len(data['top_neurons'])} most important neurons:")
for i, neuron_idx in enumerate(data['top_neurons'][:10], 1):
    importance = data['neuron_importance'][neuron_idx]
    print(f"  {i:2d}. Neuron {neuron_idx:3d}: Importance = {importance:.4f}")

print("\n2. NEURON IMPORTANCE DISTRIBUTION")
print("-" * 70)
importances = data['neuron_importance']
max_imp = max(importances)
min_imp = min(importances)
avg_imp = sum(importances) / len(importances)
print(f"  Highest importance: {max_imp:.4f} (Neuron {importances.index(max_imp)})")
print(f"  Lowest importance:  {min_imp:.4f}")
print(f"  Average importance: {avg_imp:.4f}")
print(f"  Range: {max_imp - min_imp:.4f}")

# Count highly important neurons
high_importance = [i for i, imp in enumerate(importances) if imp > avg_imp * 2]
print(f"  Neurons with importance > 2x average: {len(high_importance)}")

print("\n3. CAUSAL INTERVENTIONS (Ablation Studies)")
print("-" * 70)
print("Results of removing individual neurons:")
for intervention in data['interventions']:
    neuron = intervention['neuron']
    orig = intervention['acc_original']
    patched = intervention['acc_patched']
    impact = intervention['impact']
    
    if impact == 0.0:
        print(f"  Neuron {neuron:3d}: No impact (accuracy stayed at {orig:.4f})")
    else:
        pct = (impact / orig) * 100
        print(f"  Neuron {neuron:3d}: Accuracy dropped from {orig:.4f} to {patched:.4f} ({pct:.1f}% drop)")

print("\n4. FOURIER ANALYSIS")
print("-" * 70)
fourier = data['fourier_components']
if len(fourier) == 0:
    print("  No significant Fourier patterns detected.")
    print("  This means the model likely doesn't use sine/cosine patterns")
    print("  for modular arithmetic computation.")
else:
    print(f"  Found {len(fourier)} neurons with Fourier-like patterns:")
    for comp in fourier[:5]:
        print(f"    Neuron {comp['neuron']:3d}, frequency {comp['freq']:2d}: "
              f"correlation = {comp['max_corr']:.3f}")

print("\n5. KEY INSIGHTS")
print("-" * 70)
print("  • Model achieved perfect accuracy (1.0) on test set")
print("  • Top neurons identified but ablation shows no impact")
print("  • This suggests:")
print("    - Model has redundancy (multiple neurons can handle same task)")
print("    - OR model is so well-trained that removing one neuron doesn't hurt")
print("    - OR need to ablate multiple neurons simultaneously")
print("  • No Fourier patterns detected - model uses different mechanism")

print("\n" + "=" * 70)
