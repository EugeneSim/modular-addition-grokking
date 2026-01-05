# Glossary: ML Terms and Acronyms

This glossary explains all technical terms, acronyms, and concepts used in this project. Use this as a reference when reading the code or documentation.

## Table of Contents
- [Acronyms](#acronyms)
- [Neural Network Terms](#neural-network-terms)
- [PyTorch Terms](#pytorch-terms)
- [Training Terms](#training-terms)
- [Interpretability Terms](#interpretability-terms)
- [Mathematical Terms](#mathematical-terms)

---

## Acronyms

### **AdamW**
- **Definition**: Adaptive Moment Estimation with Weight Decay
- **Context**: An optimization algorithm used to update neural network weights during training
- **Why it matters**: More sophisticated than basic gradient descent, adapts learning rate per parameter and includes weight decay for regularization
- **Example**: `optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)`

### **GPU**
- **Definition**: Graphics Processing Unit
- **Context**: Hardware that accelerates neural network training (much faster than CPU)
- **Why it matters**: Training on GPU can be 10-100x faster than CPU
- **Example**: "Training on GPU takes 1 hour vs 10 hours on CPU"

### **LLM**
- **Definition**: Large Language Model
- **Context**: Models like GPT that understand and generate text
- **Why it matters**: This project uses similar interpretability techniques used to understand LLMs

### **MI**
- **Definition**: Mechanistic Interpretability
- **Context**: Field of research focused on understanding how neural networks work internally
- **Why it matters**: The main goal of this project - reverse-engineering how the model learned modular addition

### **MLP**
- **Definition**: Multi-Layer Perceptron
- **Context**: A type of neural network with multiple layers of neurons
- **Why it matters**: Our model (`ToyMLP`) is a simple MLP with embedding layer, hidden layer, and output layer
- **Example**: `class ToyMLP(nn.Module)` - our neural network architecture

### **ReLU**
- **Definition**: Rectified Linear Unit
- **Context**: An activation function that outputs the input if positive, otherwise 0
- **Why it matters**: Introduces non-linearity, allowing neural networks to learn complex patterns
- **Formula**: `f(x) = max(0, x)`
- **Example**: `self.relu = nn.ReLU()` - activation function in our model

---

## Neural Network Terms

### **Activation**
- **Definition**: The output value of a neuron after applying an activation function
- **Context**: What a neuron "fires" or outputs for a given input
- **Why it matters**: Activations show what the model is "thinking" internally - key for interpretability
- **Example**: After ReLU, neuron outputs 0.5 for input 0.5, but 0 for input -0.3

### **Bias**
- **Definition**: A constant value added to the weighted sum in a neuron
- **Context**: Allows neurons to shift their activation threshold
- **Why it matters**: Helps the model fit data better by adjusting decision boundaries
- **Example**: `y = weight * x + bias` - bias shifts the line up or down

### **Embedding**
- **Definition**: Converting discrete items (like integers) into dense continuous vectors
- **Context**: Our model converts input numbers (0-112) into 128-dimensional vectors
- **Why it matters**: Neural networks work better with continuous values; embeddings allow learning relationships between numbers
- **Example**: Number 5 → `[0.23, -0.45, 0.67, ...]` (128 numbers)

### **Forward Pass**
- **Definition**: The process of passing input data through the network to get predictions
- **Context**: Input → Embedding → Hidden Layer → Output
- **Why it matters**: This is how the model makes predictions
- **Example**: `outputs = model(inputs)` - this performs a forward pass

### **Gradient**
- **Definition**: The derivative of the loss function with respect to each weight
- **Context**: Tells us how much to adjust each weight to reduce loss
- **Why it matters**: Used in backpropagation to update weights and improve the model
- **Example**: If gradient is positive, decreasing the weight reduces loss

### **Hidden Layer**
- **Definition**: A layer of neurons between input and output layers
- **Context**: Our model has one hidden layer with 256 neurons
- **Why it matters**: Allows the model to learn complex, non-linear patterns
- **Example**: `self.fc1 = nn.Linear(256, 256)` - our hidden layer

### **Loss Function**
- **Definition**: A function that measures how wrong the model's predictions are
- **Context**: We use CrossEntropyLoss to compare predictions to correct answers
- **Why it matters**: The model tries to minimize loss during training
- **Example**: Lower loss = better predictions

### **Neuron**
- **Definition**: A basic computational unit in a neural network
- **Context**: Each neuron takes inputs, computes weighted sum, applies activation function
- **Why it matters**: The building blocks of neural networks
- **Example**: Our hidden layer has 256 neurons

### **Weight**
- **Definition**: A parameter that determines how much influence an input has
- **Context**: Each connection between neurons has a weight that gets adjusted during training
- **Why it matters**: These are what the model "learns" - the values that enable it to solve the task
- **Example**: ~50,000 weights in our model that get updated during training

---

## PyTorch Terms

### **DataLoader**
- **Definition**: PyTorch utility that batches and shuffles data for training
- **Context**: `DataLoader(train_dataset, batch_size=64)` - processes data in batches
- **Why it matters**: Efficiently feeds data to the model during training
- **Example**: Instead of processing one example at a time, processes 64 at once

### **Module**
- **Definition**: Base class for all neural network components in PyTorch
- **Context**: `class ToyMLP(nn.Module)` - our model inherits from Module
- **Why it matters**: Provides standard interface and functionality for neural networks
- **Example**: All PyTorch models inherit from `nn.Module`

### **Tensor**
- **Definition**: A multi-dimensional array (like NumPy arrays but for GPU)
- **Context**: All data in PyTorch is stored as tensors
- **Why it matters**: Enables efficient computation on GPU
- **Example**: `torch.tensor([[5, 10], [20, 30]])` - a 2D tensor with 2 examples

### **TensorDataset**
- **Definition**: PyTorch class that wraps tensors into a dataset
- **Context**: `TensorDataset(inputs, labels)` - creates dataset from tensors
- **Why it matters**: Converts raw data into format DataLoader can use
- **Example**: Wraps input-output pairs for training

---

## Training Terms

### **Batch**
- **Definition**: A group of examples processed together during training
- **Context**: We use batch_size=64, so 64 examples processed at once
- **Why it matters**: More efficient than processing one example at a time
- **Example**: Instead of 10,000 individual updates, we do ~156 batch updates per epoch

### **Checkpoint**
- **Definition**: A saved snapshot of the model's state at a specific point in training
- **Context**: Saved every 2000 epochs to `checkpoints/model_epoch_2000.pth`
- **Why it matters**: Allows resuming training if interrupted, or loading specific model versions
- **Example**: Can resume from epoch 2000 if training stops

### **Epoch**
- **Definition**: One complete pass through the entire training dataset
- **Context**: We train for 5000 epochs - the model sees all training data 5000 times
- **Why it matters**: More epochs = more learning, but also more time
- **Example**: Epoch 1: model sees all 10,000 examples once

### **Learning Rate**
- **Definition**: How big of steps the optimizer takes when updating weights
- **Context**: `lr=1e-3` means small, careful steps
- **Why it matters**: Too high = unstable training, too low = very slow learning
- **Example**: Learning rate of 0.001 means weights change by 0.001 * gradient

### **Overfitting**
- **Definition**: Model memorizes training data but fails on new data
- **Context**: Early in training, model overfits (high train accuracy, low test accuracy)
- **Why it matters**: Shows model hasn't learned the general rule, just memorized examples
- **Example**: 100% train accuracy but 20% test accuracy = overfitting

### **Weight Decay**
- **Definition**: A regularization technique that penalizes large weights
- **Context**: `weight_decay=1e-2` encourages smaller weights
- **Why it matters**: Helps prevent overfitting and encourages simpler solutions (important for grokking!)
- **Example**: Forces model to find simpler patterns rather than memorizing

---

## Interpretability Terms

### **Ablation**
- **Definition**: Removing or zeroing out a component to measure its importance
- **Context**: We ablate neurons (set to 0) to see if accuracy drops
- **Why it matters**: Shows which neurons are actually important for the task
- **Example**: Zeroing out neuron 42 drops accuracy by 30% → neuron 42 is important

### **Activation Analysis**
- **Definition**: Studying what values neurons output for different inputs
- **Context**: We record activations for many inputs to find patterns
- **Why it matters**: Reveals what patterns neurons detect
- **Example**: Neuron 10 always activates strongly for inputs where sum > 100

### **Circuit**
- **Definition**: A subnetwork of neurons that work together to perform a specific computation
- **Context**: We're looking for the "circuit" that computes modular addition
- **Why it matters**: Understanding circuits = understanding how the model works
- **Example**: Neurons 5, 12, and 23 form a circuit that handles carry operations

### **Causal Intervention**
- **Definition**: Directly manipulating the model to see what happens
- **Context**: Ablating neurons is a causal intervention - we cause a change and measure effect
- **Why it matters**: Shows cause-and-effect relationships, not just correlations
- **Example**: Removing neuron X causes accuracy drop → neuron X is causally important

### **Heatmap**
- **Definition**: A visual representation where colors represent values
- **Context**: We create heatmaps showing neuron activations across different inputs
- **Why it matters**: Makes patterns visible that are hard to see in raw numbers
- **Example**: Red = high activation, blue = low activation

---

## Mathematical Terms

### **Modular Arithmetic**
- **Definition**: Arithmetic where numbers "wrap around" at a certain value (modulus)
- **Context**: `(a + b) mod p` - if result exceeds p, subtract p until it's less than p
- **Why it matters**: This is the task our model learns
- **Example**: `(100 + 50) mod 113 = 37` because 150 - 113 = 37

### **Prime Number**
- **Definition**: A number divisible only by 1 and itself
- **Context**: We use p=113 (a prime) to avoid simple patterns
- **Why it matters**: If p=100, model might learn "just add last digits" - primes prevent this
- **Example**: 113 is prime, so no simple shortcuts exist

### **Logits**
- **Definition**: Raw output scores from the model before converting to probabilities
- **Context**: Our model outputs 113 logits (one for each possible answer 0-112)
- **Why it matters**: Higher logit = model is more confident that's the answer
- **Example**: Logits `[0.1, 0.5, 2.3, ...]` → model thinks answer is 2 (highest logit)

---

## Project-Specific Terms

### **Grokking**
- **Definition**: Sudden transition from memorization to generalization after prolonged training
- **Context**: Model suddenly "gets it" - test accuracy jumps from 20% to 90% in a few epochs
- **Why it matters**: This is the main phenomenon we're studying
- **Example**: Epoch 5000: 20% test accuracy → Epoch 5100: 90% test accuracy = grokking!

### **Memorization Phase**
- **Definition**: Early training phase where model memorizes examples but doesn't generalize
- **Context**: Epochs 0-3000: high train accuracy, low test accuracy
- **Why it matters**: Shows model hasn't learned the algorithm yet
- **Example**: Model remembers (5,10)→15 but fails on similar unseen pairs

### **Generalization Phase**
- **Definition**: After grokking, model understands the rule and works on new data
- **Context**: Epochs 3000+: high accuracy on both train and test sets
- **Why it matters**: Model truly learned the algorithm, not just memorization
- **Example**: Model can handle any (a,b) pair, even if never seen during training

---

## Quick Reference: Common Patterns

**Training Loop Pattern:**
```
for epoch in range(epochs):
    for batch in dataloader:
        outputs = model(inputs)      # Forward pass
        loss = criterion(outputs, targets)  # Compute loss
        loss.backward()              # Backward pass (compute gradients)
        optimizer.step()              # Update weights
```

**Model Architecture Pattern:**
```
Input → Embedding → Hidden Layer → Output Layer → Predictions
```

**Interpretability Pattern:**
```
Load Model → Get Activations → Identify Important Neurons → Ablate → Measure Impact
```

---

## Additional Resources

- **PyTorch Documentation**: https://pytorch.org/docs/
- **Neural Networks Basics**: https://neuralnetworksanddeeplearning.com/
- **Mechanistic Interpretability**: https://neelnanda.io/mechanistic-interpretability

---

*This glossary is designed to be comprehensive but accessible. If you encounter a term not listed here, please check the code comments or documentation for context-specific explanations.*
