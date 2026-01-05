# Modular Addition Grokking: Mechanistic Interpretability of a Toy Neural Network

This project is a hands-on exploration of the "grokking" phenomenon in neural networks, using a simple algorithmic task: modular addition (e.g., predicting `(a + b) mod p`). The project trains a small neural network on this task, observes grokking (a sudden shift from memorization to generalization after prolonged training), and then applies mechanistic interpretability techniques to reverse-engineer the model's internal computations.

Disclaimer: This was done with the help of AI tools 

## Table of Contents

- [Project Overview](#project-overview)
- [Workflow Overview](#workflow-overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Project Structure](#project-structure)
- [Code Flow](#code-flow)
- [Expected Results](#expected-results)
- [Hardware Requirements](#hardware-requirements)
- [Troubleshooting](#troubleshooting)
- [Glossary](#glossary)
- [References](#references)

---

## Project Overview

### What Is Grokking?

**Grokking** is a fascinating phenomenon where neural networks initially memorize training data (high train accuracy, low test accuracy) but then suddenly "grok" the underlying algorithm after prolonged training, achieving high accuracy on both training and test sets.

### What This Project Does

1. **Trains a neural network** to perform modular addition: `(a + b) mod 113`
2. **Observes grokking** - the sudden transition from memorization to generalization
3. **Analyzes the model** using mechanistic interpretability to understand how it learned

### Learning Goals

- Gain intuition for neural network training dynamics (overfitting vs. generalization)
- PyTorch proficiency
- Interpretability methods like activation probing and causal interventions
- Research-oriented AI concepts

### Key Concepts

- **Grokking**: Models initially overfit but suddenly generalize after many epochs
- **Mechanistic Interpretability (MI)**: Probing model internals to find "circuits" (subnetworks handling specific tasks)
- **Applications**: Ties to frontier AI (e.g., understanding LLMs like GPT for safety/alignment)

---

## Workflow Overview

### High-Level Workflow Diagram

```mermaid
graph LR
    A[dataset.py<br/>Generate Data] --> B[train.py<br/>Train Model]
    B --> C[checkpoints/<br/>Saved Model]
    C --> D[interpretability.py<br/>Analyze Model]
    D --> E[results/<br/>Visualizations]
    
    style A fill:#e1f5ff
    style B fill:#fff4e1
    style C fill:#e8f5e9
    style D fill:#f3e5f5
    style E fill:#fce4ec
```

### Step-by-Step Workflow

1. **Data Generation** (`dataset.py`)
   - Generates all possible `(a, b)` pairs for modular addition
   - Computes labels: `(a + b) mod 113`
   - Splits into train (80%) and test (20%) sets
   - Creates DataLoaders for batch processing

2. **Model Training** (`train.py`)
   - Initializes neural network (ToyMLP)
   - Trains for 1000 epochs (~1 hour on average laptop)
   - Monitors training loss and test accuracy
   - Detects grokking moment (sudden accuracy jump)
   - Saves checkpoints and final model

3. **Interpretability Analysis** (`interpretability.py`)
   - Loads trained model from checkpoint
   - Identifies important neurons
   - Creates activation heatmaps
   - Performs causal interventions (neuron ablation)
   - Tests Fourier hypothesis
   - Saves results and visualizations

---

## Installation

### Prerequisites

- **Python 3.8+** (check with `python --version`)
- **16GB RAM** (recommended, 8GB minimum)
- **Internet connection** (for downloading packages)

### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/modular-addition-grokking.git
   cd modular-addition-grokking
   ```

2. **Create virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # Windows:
   venv\Scripts\activate
   
   # Mac/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python test_setup.py
   ```

   Expected output: All tests should pass ✓

For detailed installation instructions, see [EXECUTION_GUIDE_DETAILED.md](EXECUTION_GUIDE_DETAILED.md).

---

## Quick Start

### Minimal Example

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify setup
python test_setup.py

# 3. Train model (~1 hour)
python train.py

# 4. Analyze results (~15 minutes)
python interpretability.py
```

### Quick Demo (Faster, Smaller)

For a quick demonstration with smaller parameters:

```bash
python example_quick_start.py
```

This trains a smaller model for 5000 epochs (may or may not show grokking - uses different parameters).

---

## Detailed Usage

### Step 1: Dataset Generation

The dataset is generated automatically during training, but it can be tested:

```bash
python dataset.py
```

**What happens:**
- Generates all possible `(a, b)` pairs where `a, b ∈ [0, 112]`
- Computes `(a + b) mod 113` for each pair
- Creates 12,769 total examples
- Splits into train (10,000) and test (2,000) sets

**Output:**
```
Dataset generated with p=113
Train batches: 157, Test batches: 32
Sample: tensor([5, 10]) -> tensor(15)
```

### Step 2: Train the Model

```bash
python train.py
```

**What happens:**
- Trains ToyMLP for 1000 epochs (~1 hour on average laptop)
- Saves checkpoints every 500 epochs
- Prints test accuracy every 50 epochs
- Generates `training_curves.png` on completion

**Expected Output:**
```
Generating dataset...
Starting training for 1000 epochs...
Model parameters: 50,369
Epoch     0: Train Loss 4.5234, Test Acc 0.0089
Epoch    50: Train Loss 0.1234, Test Acc 0.1234
Epoch   500: Train Loss 0.0023, Test Acc 0.8543
  ⚡ Potential grokking detected! Accuracy jumped from 0.1234 to 0.8543
...
Epoch  1000: Train Loss 0.0001, Test Acc 0.9876
Training complete! Final model saved: checkpoints/model_final.pth
```

**Hyperparameters:**
- `p=113`: Modulo prime (avoids simple patterns)
- `embed_dim=128`: Embedding dimension
- `hidden_dim=256`: Hidden layer size
- `epochs=1000`: Training epochs (~1 hour)
- `lr=1e-3`: Learning rate
- `weight_decay=1e-2`: Weight decay (helps grokking)
- `batch_size=64`: Batch size

**Expected Behavior:**
- **Memorization Phase** (epochs 0-500): High train accuracy (~100%), low test accuracy (~10-30%)
- **Grokking Phase** (epochs 500-800): Sudden jump in test accuracy (grokking!)
- **Generalization Phase** (epochs 800+): High accuracy on both sets (~95%+)

### Step 3: Interpretability Analysis

After training completes:

```bash
python interpretability.py
```

**What happens:**
1. **Activation Analysis**: Identifies most active neurons
2. **Activation Heatmaps**: Visualizes neuron activations across inputs
3. **Causal Interventions**: Ablates neurons to measure their importance
4. **Fourier Analysis**: Tests hypothesis that model uses Fourier-like embeddings

**Output Files:**
- `results/activation_heatmap.png`: Visualization of neuron activations
- `results/analysis_summary.json`: Detailed analysis results

**Expected Output:**
```
============================================================
MECHANISTIC INTERPRETABILITY ANALYSIS
============================================================
Model loaded from checkpoints/model_final.pth
Identifying important neurons...

Top 20 most active neurons:
  Neuron  42: importance = 0.1234
  Neuron  87: importance = 0.1156
  ...

Activation heatmap saved to results/activation_heatmap.png

CAUSAL INTERVENTIONS
Performing causal intervention: ablating neuron 42...
Original accuracy: 0.9876
Patched accuracy: 0.7234
Impact: 0.2642 (26.75% relative drop)
...
```

---

## Project Structure

```
modular-addition-grokking/
├── dataset.py                    # Dataset generation
├── model.py                      # Neural network model definition
├── train.py                      # Training script
├── interpretability.py           # Mechanistic interpretability analysis
├── example_quick_start.py        # Quick demo script
├── test_setup.py                 # Setup verification
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── EXECUTION_GUIDE_DETAILED.md   # Detailed step-by-step guide
├── GLOSSARY.md                   # ML terms and acronyms reference
├── CONCLUSION.md                 # Project results and interpretation
├── checkpoints/                  # Saved model checkpoints
│   ├── model_epoch_xxxx.pth
│   ├── model_epoch_xxxx.pth
│   └── model_final.pth
└── results/                      # Analysis results and plots
    ├── activation_heatmap.png
    └── analysis_summary.json
```

---

## Code Flow

### How Files Connect

```
dataset.py
    │
    ├─> Generates train_loader, test_loader
    │
    └─> Used by train.py and interpretability.py
         │
         │
train.py
    │
    ├─> Imports: dataset.py, model.py
    │
    ├─> Generates dataset using dataset.py
    │
    ├─> Creates model using model.py
    │
    ├─> Trains model (1000 epochs)
    │
    ├─> Saves checkpoints to checkpoints/
    │
    └─> Generates training_curves.png
         │
         │
interpretability.py
    │
    ├─> Imports: model.py, dataset.py
    │
    ├─> Loads model from checkpoints/model_final.pth
    │
    ├─> Generates test dataset using dataset.py
    │
    ├─> Analyzes activations using model.get_activations()
    │
    ├─> Performs causal interventions
    │
    └─> Saves results to results/
```

### What Each File Does

**`dataset.py`**
- Generates all possible `(a, b)` pairs for modular addition
- Computes labels: `(a + b) mod 113`
- Splits into train/test sets
- Creates PyTorch DataLoaders for batch processing

**`model.py`**
- Defines `ToyMLP` class (neural network architecture)
- Implements forward pass: Input → Embedding → Hidden Layer → Output
- Provides `get_activations()` method for interpretability

**`train.py`**
- Orchestrates training process
- Generates dataset, initializes model, sets up optimizer
- Runs training loop: Forward → Loss → Backward → Update
- Monitors progress and detects grokking
- Saves checkpoints and generates plots

**`interpretability.py`**
- Loads trained model
- Analyzes internal activations
- Identifies important neurons
- Performs causal interventions (ablation studies)
- Tests Fourier hypothesis
- Generates visualizations

---

## Expected Results

### Training Dynamics

The model typically shows three phases:

1. **Memorization Phase** (epochs 0-500)
   - Train accuracy: ~100%
   - Test accuracy: ~10-30%
   - Model memorizes training examples but doesn't generalize

2. **Grokking Phase** (epochs 500-800)
   - Sudden jump in test accuracy (20% → 90%+)
   - Model "discovers" the underlying algorithm
   - This is the grokking moment!

3. **Generalization Phase** (epochs 800+)
   - Train accuracy: ~98%+
   - Test accuracy: ~95%+
   - Model truly understands modular addition

### Interpretability Findings

Typical results from analysis:

- **Top Neurons**: 5-10 neurons handle most of the computation
- **Causal Impact**: Ablating top neurons drops accuracy by 20-40%
- **Fourier Patterns**: Some neurons may show correlation with sine/cosine patterns
- **Circuit Discovery**: Small subset of neurons form a "circuit" for modular addition

### Visualizations

**`training_curves.png`**
- Left plot: Training loss over time (decreasing)
- Right plot: Test accuracy over time (shows grokking jump)

**`results/activation_heatmap.png`**
- Heatmap showing neuron activations across different inputs
- Colors represent activation strength (red = high, blue = low)

---

## Hardware Requirements

### Minimum Requirements

- **CPU**: Any modern processor (Intel i5 or equivalent)
- **RAM**: 8GB (16GB recommended)
- **Storage**: 500MB free space
- **GPU**: Optional but recommended (10x faster training)

### Expected Training Times

**With CPU:**
- 1000 epochs: ~2-3 hours

**With GPU (NVIDIA, CUDA-enabled):**
- 1000 epochs: ~30-60 minutes

**Average Laptop (16GB RAM, integrated GPU):**
- 1000 epochs: ~1 hour

### Memory Usage

- **Training**: ~2-4GB RAM
- **Analysis**: ~1-2GB RAM
- **Checkpoints**: ~10-20MB per checkpoint

---

## Troubleshooting

### Common Issues

**No grokking observed:**
- Train longer: Increase epochs to 8000-10000
- Lower learning rate: Change `lr=1e-3` to `lr=5e-4` in `train.py`
- Try different prime: Change `p=113` to `p=97` in `train.py`
- Increase weight decay: Change `weight_decay=1e-2` to `weight_decay=2e-2`

**Training takes too long:**
- Already optimized for ~1 hour (1000 epochs)
- Use GPU if available (much faster)
- Reduce model size: `embed_dim=96`, `hidden_dim=192`

**Out of memory:**
- Reduce batch size: Change `batch_size=64` to `batch_size=32` in `dataset.py`
- Use CPU instead of GPU
- Close other applications

**Import errors:**
- Run `pip install -r requirements.txt`
- Ensure virtual environment is activated
- Check Python version: `python --version` (need 3.8+)

For detailed troubleshooting, see [EXECUTION_GUIDE_DETAILED.md](EXECUTION_GUIDE_DETAILED.md).

---

## Glossary

This project uses many ML-specific terms. For a comprehensive glossary, see [GLOSSARY.md](GLOSSARY.md).

**Quick Reference:**
- **Grokking**: Sudden transition from memorization to generalization
- **Epoch**: One complete pass through training data
- **Activation**: Output value of a neuron
- **Ablation**: Removing/zeroing out a component to measure importance
- **Embedding**: Converting discrete items to continuous vectors
- **MLP**: Multi-Layer Perceptron (type of neural network)

---

## References

### Papers

- Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). **Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets**. *arXiv preprint arXiv:2201.02177*.
  - Focus on Sections 1-3 for understanding the phenomenon

### Resources

- **Mechanistic Interpretability**: [Neel Nanda's MI Quickstart](https://neelnanda.io/mechanistic-interpretability)
  - Focus on "Toy Models" section
- **TransformerLens**: [GitHub Repository](https://github.com/neelnanda-io/TransformerLens)
- **PyTorch Documentation**: https://pytorch.org/docs/

### Optional Reading

- YouTube: "What is Grokking in Neural Networks?"
- Blog posts on mechanistic interpretability
- Papers on neural network interpretability

---

## Extensions

Potential extensions to explore:

1. **Vary `p`**: Test with non-prime numbers or different primes
2. **Transformer Architecture**: Upgrade to full transformer with attention
3. **Noise Robustness**: Add noise to inputs and test grokking persistence
4. **Circuit Sparsity**: Compute fraction of neurons needed for task
5. **Other Operations**: Extend to modular multiplication
6. **Visualization**: Create interactive visualizations of activations
7. **Ablation Studies**: Systematic ablation of different components

---

## License

This project is for educational purposes. Feel free to use and modify as needed.

---

## Contributing

This is a learning project. Feel free to:
- Report issues
- Suggest improvements
- Share your findings
- Extend the codebase

---

## Acknowledgments

This project demonstrates research-oriented AI concepts and is inspired by work from:
- Anthropic
- OpenAI
- The mechanistic interpretability community
- Neel Nanda's work on TransformerLens

---

**Happy grokking! 🚀**

## Documentation

- **[EXECUTION_GUIDE_DETAILED.md](EXECUTION_GUIDE_DETAILED.md)**: Detailed step-by-step instructions with zero assumptions
- **[GLOSSARY.md](GLOSSARY.md)**: Comprehensive ML terms and acronyms reference
- **[CONCLUSION.md](CONCLUSION.md)**: Project results, findings, and interpretation

---

## Project Status

✅ **Complete** - Model achieved 100% test accuracy and full interpretability analysis completed.

See [CONCLUSION.md](CONCLUSION.md) for detailed results and insights.
