# Detailed Execution Guide: Modular Addition Grokking

This guide provides step-by-step instructions with zero assumptions. Every command, every step, and every expected outcome is explained in detail.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Step 1: Environment Setup](#step-1-environment-setup)
3. [Step 2: Verify Installation](#step-2-verify-installation)
4. [Step 3: Understand Dataset Generation](#step-3-understand-dataset-generation)
5. [Step 4: Train the Model](#step-4-train-the-model)
6. [Step 5: Analyze Results](#step-5-analyze-results)
7. [Troubleshooting](#troubleshooting)
8. [Platform-Specific Notes](#platform-specific-notes)

---

## Prerequisites

### What You Need

1. **Python 3.8 or higher**
   - Check your version: Open terminal/command prompt and type: `python --version`
   - If you see "Python 3.8.x" or higher, you're good
   - If not, download from https://www.python.org/downloads/
   - **Important**: During installation, check "Add Python to PATH"

2. **16GB RAM** (recommended, but 8GB minimum)
   - Check on Windows: Task Manager → Performance → Memory
   - Check on Mac: Apple menu → About This Mac
   - Check on Linux: `free -h` in terminal

3. **Internet connection** (for downloading packages)

4. **Text editor or IDE** (optional but helpful)
   - VS Code, PyCharm, or even Notepad++

### What This Project Does

- **Task**: Teaches a neural network to do modular addition: `(a + b) mod 113`
- **Goal**: Observe "grokking" - when the model suddenly learns the algorithm
- **Analysis**: Understand how the model learned internally

---

## Step 1: Environment Setup

### 1.1 Open Terminal/Command Prompt

**Windows:**
- Press `Win + R`, type `cmd`, press Enter
- OR Press `Win + X`, select "Windows PowerShell" or "Command Prompt"

**Mac:**
- Press `Cmd + Space`, type "Terminal", press Enter
- OR Applications → Utilities → Terminal

**Linux:**
- Press `Ctrl + Alt + T` (most distributions)
- OR find Terminal in applications menu

### 1.2 Navigate to Project Directory

**What this means**: You need to be in the folder containing the project files.

**Command:**
```bash
cd path/to/modular-addition-grokking
```

**Example (Windows):**
```bash
cd C:\Users\YourName\Documents\modular-addition-grokking
```

**Example (Mac/Linux):**
```bash
cd ~/Documents/modular-addition-grokking
```

**How to find your path:**
- Windows: Right-click folder → Properties → Copy "Location"
- Mac: Right-click folder → Get Info → Copy path
- Linux: Right-click folder → Properties → Copy path

**Verify you're in the right place:**
```bash
dir
```
(Windows) or
```bash
ls
```
(Mac/Linux)

You should see files like `dataset.py`, `train.py`, `requirements.txt`

### 1.3 Create Virtual Environment (Recommended)

**What is this?** A virtual environment isolates this project's packages from other Python projects.

**Why do this?** Prevents conflicts between different projects' package versions.

**Command:**
```bash
python -m venv venv
```

**What happens:** Creates a folder called `venv` in your project directory.

**Activate it:**

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

**How to know it worked:** Your prompt should show `(venv)` at the beginning:
```
(venv) C:\Users\...\modular-addition-grokking>
```

**To deactivate later:** Just type `deactivate`

### 1.4 Install Required Packages

**What are packages?** Pre-written code libraries that provide functionality (like PyTorch for neural networks).

**Command:**
```bash
pip install -r requirements.txt
```

**What this does:** Reads `requirements.txt` and installs all listed packages.

**Expected output:**
```
Collecting torch...
Downloading torch-2.x.x...
Installing collected packages...
Successfully installed torch-2.x.x numpy-1.x.x ...
```

**Time:** 5-15 minutes depending on internet speed.

**If you get errors:**
- "pip not found": Try `python -m pip install -r requirements.txt`
- "Permission denied": On Mac/Linux, try `sudo pip install -r requirements.txt`
- "No module named pip": Install pip first: `python -m ensurepip --upgrade`

**Verify installation:**
```bash
python -c "import torch; print(torch.__version__)"
```

Should print something like: `2.0.0` or `2.1.0`

---

## Step 2: Verify Installation

### 2.1 Run Test Script

**Purpose:** Verify everything is set up correctly before training.

**Command:**
```bash
python test_setup.py
```

**Expected output:**
```
============================================================
SETUP VERIFICATION TEST
============================================================
Testing imports...
✓ Core packages imported successfully

Testing dataset generation...
✓ Dataset generated: p=13, train batches=7, test batches=2

Testing model...
✓ Model created: input shape torch.Size([2, 2]), output shape torch.Size([2, 13])
  Model parameters: 2,469

Testing training step...
✓ Training step successful: loss=2.3456

============================================================
SUMMARY
============================================================
Imports        : ✓ PASS
Dataset        : ✓ PASS
Model          : ✓ PASS
Training       : ✓ PASS

✓ All tests passed! Setup is ready.
```

**If you see errors:**
- See [Troubleshooting](#troubleshooting) section below
- Common issue: Missing packages → Re-run `pip install -r requirements.txt`

---

## Step 3: Understand Dataset Generation

### 3.1 What Happens During Dataset Generation

**Purpose:** Create training and test data for modular addition.

**What the code does:**
1. Generates all possible pairs `(a, b)` where `a` and `b` are 0-112
2. Computes correct answers: `(a + b) mod 113`
3. Splits into train (80%) and test (20%)
4. Creates batches for efficient processing

**Example:**
- Input: `(5, 10)` → Output: `15` (because 5 + 10 = 15)
- Input: `(100, 50)` → Output: `37` (because 150 mod 113 = 37)

### 3.2 Test Dataset Generation (Optional)

**Command:**
```bash
python dataset.py
```

**Expected output:**
```
Dataset generated with p=113
Train batches: 157, Test batches: 32
Sample batch shape: torch.Size([64, 2])
Sample: tensor([5, 10]) -> tensor(15) (should be 15)
```

**What this confirms:**
- Dataset generation works
- Batch size is correct (64 examples per batch)
- Labels are computed correctly

**You don't need to run this** - it happens automatically during training, but it's good to verify.

---

## Step 4: Train the Model

### 4.1 Understanding What Training Does

**The Training Process:**
1. **Forward Pass**: Model makes predictions
2. **Loss Calculation**: Measures how wrong predictions are
3. **Backward Pass**: Calculates how to adjust weights
4. **Weight Update**: Updates model to improve predictions
5. **Repeat**: 1000 times (epochs)

**What You'll See:**
- Model starts with random guesses (~1% accuracy)
- Gradually improves on training data
- Test accuracy stays low initially (memorization)
- Suddenly jumps to high accuracy (grokking!)

### 4.2 Start Training

**Command:**
```bash
python train.py
```

**What Happens:**
1. Creates `checkpoints/` folder (if it doesn't exist)
2. Generates dataset
3. Initializes model (~50,000 parameters)
4. Starts training loop

**Expected Initial Output:**
```
Generating dataset...
Starting training for 1000 epochs...
Model parameters: 50,369
Epoch     0: Train Loss 4.5234, Test Acc 0.0089
Epoch    50: Train Loss 0.1234, Test Acc 0.1234
Epoch   100: Train Loss 0.0456, Test Acc 0.1456
...
```

### 4.3 Monitoring Training

**What to Watch For:**

1. **Training Loss**: Should decrease over time
   - Starts high (~4-5)
   - Decreases gradually
   - Eventually very low (<0.01)

2. **Test Accuracy**: This is the key metric
   - Starts very low (~1-10%)
   - Stays low for many epochs (memorization phase)
   - **Grokking moment**: Sudden jump to 80-95%+
   - Usually happens around epoch 500-800

3. **Grokking Detection**: Script will print:
   ```
   ⚡ Potential grokking detected! Accuracy jumped from 0.1234 to 0.8543
   ```

**Time Estimate:**
- **CPU**: ~2-3 hours for 1000 epochs
- **GPU**: ~30-60 minutes for 1000 epochs
- **Average laptop**: ~1 hour

### 4.4 Checkpoints

**What are checkpoints?** Saved snapshots of the model at specific epochs.

**When saved:** Every 500 epochs automatically

**Where saved:** `checkpoints/model_epoch_500.pth`, `checkpoints/model_epoch_1000.pth`, etc.

**Why useful:** 
- Can resume training if interrupted
- Can load specific model versions
- Final model saved as `checkpoints/model_final.pth`

### 4.5 Training Completion

**When it finishes:**
```
Training complete! Final model saved: checkpoints/model_final.pth
Training curves saved to training_curves.png

Training summary:
Final test accuracy: 0.9876
Final training loss: 0.0001
```

**What you get:**
- `checkpoints/model_final.pth`: Trained model
- `training_curves.png`: Visualization of training progress

**If training stops early:**
- Check error messages
- See [Troubleshooting](#troubleshooting)
- You can resume from last checkpoint (advanced)

---

## Step 5: Analyze Results

### 5.1 View Training Curves

**File:** `training_curves.png`

**What it shows:**
- Left plot: Training loss over time (should decrease)
- Right plot: Test accuracy over time (should show grokking jump)

**How to view:**
- Double-click the file (opens in default image viewer)
- OR open in any image viewer

**What to look for:**
- Loss curve: Smoothly decreasing
- Accuracy curve: Flat then sudden jump (grokking!)

### 5.2 Run Interpretability Analysis

**Purpose:** Understand how the model learned internally.

**Command:**
```bash
python interpretability.py
```

**What This Does:**
1. Loads the trained model
2. Analyzes which neurons are most important
3. Creates activation heatmaps
4. Performs causal interventions (ablates neurons)
5. Tests Fourier hypothesis

**Expected Output:**
```
============================================================
MECHANISTIC INTERPRETABILITY ANALYSIS
============================================================
Model loaded from checkpoints/model_final.pth
Analyzing activations...
Identifying important neurons...

Top 20 most active neurons:
  Neuron  42: importance = 0.1234
  Neuron  87: importance = 0.1156
  ...

Generating activation heatmap for relu_post...
Activation heatmap saved to results/activation_heatmap.png

============================================================
CAUSAL INTERVENTIONS
============================================================
Performing causal intervention: ablating neuron 42...
Original accuracy: 0.9876
Patched accuracy: 0.7234
Impact: 0.2642 (26.75% relative drop)
...

============================================================
FOURIER COMPONENT ANALYSIS
============================================================
Analyzing potential Fourier components...
Found 5 neurons with significant Fourier-like patterns:
  Neuron  42, freq= 1: max_corr=0.456
  ...

Analysis complete! Results saved to results/
```

**Time:** 10-30 minutes depending on hardware

**Output Files:**
- `results/activation_heatmap.png`: Visual representation of neuron activations
- `results/analysis_summary.json`: Detailed numerical results

### 5.3 Understanding the Results

**Important Neurons:**
- Top neurons are the ones doing most of the computation
- Usually 5-10 neurons handle most of the task

**Causal Interventions:**
- Shows which neurons are actually important
- High impact = neuron is crucial for the task

**Fourier Analysis:**
- Tests if model uses mathematical patterns (sine/cosine)
- Significant correlations suggest sophisticated computation

---

## Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'torch'"

**Solution:**
```bash
pip install -r requirements.txt
```

**If that doesn't work:**
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Problem: "CUDA out of memory" or "Out of memory"

**What this means:** Your GPU/RAM ran out of memory.

**Solutions:**
1. Reduce batch size in `train.py`:
   - Find `batch_size=64` in `dataset.py`
   - Change to `batch_size=32` or `batch_size=16`

2. Use CPU instead of GPU:
   - Training will be slower but uses less memory
   - PyTorch will automatically use CPU if GPU unavailable

3. Close other applications to free RAM

### Problem: Training takes too long

**Solutions:**
1. Reduce epochs (already set to 1000 for ~1 hour)
2. Use smaller model:
   - In `train.py`, change `embed_dim=128` to `embed_dim=64`
   - Change `hidden_dim=256` to `hidden_dim=128`

3. Use GPU if available (much faster)

### Problem: No grokking observed

**What this means:** Test accuracy never jumps to high values.

**Solutions:**
1. Train longer: Increase epochs to 8000 or 10000
2. Lower learning rate: Change `lr=1e-3` to `lr=5e-4` in `train.py`
3. Try different prime: Change `p=113` to `p=97` in `train.py`
4. Increase weight decay: Change `weight_decay=1e-2` to `weight_decay=2e-2`

**Note:** Grokking is not guaranteed - it depends on many factors. The model may still learn well without a dramatic "grokking moment."

### Problem: "FileNotFoundError: checkpoints/model_final.pth"

**What this means:** You're trying to analyze before training completes.

**Solution:** Run `python train.py` first and wait for it to finish.

### Problem: Import errors with transformer-lens

**What this means:** transformer-lens package failed to install.

**Solution:** It's optional! The interpretability code works without it. You can skip installing it or install manually:
```bash
pip install transformer-lens
```

### Problem: Python command not found

**Windows:**
- Try `py` instead of `python`
- OR `python3` instead of `python`

**Mac/Linux:**
- Try `python3` instead of `python`
- May need to install Python: `brew install python3` (Mac) or use package manager (Linux)

### Problem: Permission denied errors

**Mac/Linux:**
- Use `sudo` (not recommended for virtual environments)
- Better: Make sure you activated virtual environment
- Check file permissions: `chmod +x script.py`

**Windows:**
- Run command prompt as Administrator
- OR check antivirus isn't blocking

---

## Platform-Specific Notes

### Windows

**Path Separators:**
- Use backslashes: `C:\Users\...`
- Or forward slashes work too: `C:/Users/...`

**Python Command:**
- May need to use `py` instead of `python`
- Check: `py --version`

**Long Paths:**
- If you get "path too long" errors, move project closer to root (e.g., `C:\projects\`)

### Mac

**Python Version:**
- System Python may be 2.7 (old)
- Use `python3` command
- Install via Homebrew: `brew install python3`

**Permissions:**
- May need to allow terminal in System Preferences → Security & Privacy

### Linux

**Package Manager:**
- Install Python: `sudo apt install python3 python3-pip` (Ubuntu/Debian)
- OR `sudo yum install python3 python3-pip` (RedHat/CentOS)

**Virtual Environment:**
- May need: `sudo apt install python3-venv`

---

## Quick Reference: Common Commands

```bash
# Navigate to project
cd path/to/modular-addition-grokking

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install packages
pip install -r requirements.txt

# Test setup
python test_setup.py

# Train model
python train.py

# Analyze results
python interpretability.py

# Deactivate virtual environment (when done)
deactivate
```

---

## Next Steps After Completion

1. **Experiment with parameters:**
   - Try different `p` values (97, 127, etc.)
   - Adjust model size
   - Change learning rate

2. **Extend the project:**
   - Try modular multiplication instead of addition
   - Upgrade to transformer architecture
   - Add noise to test robustness

3. **Document your findings:**
   - Note when grokking occurred
   - Record which neurons were most important
   - Share results!

---

## Getting Help

If you encounter issues not covered here:

1. Check error messages carefully - they often tell you what's wrong
2. Verify all prerequisites are met
3. Ensure you're in the correct directory
4. Try running commands one at a time to isolate the problem
5. Check that virtual environment is activated (if using one)

---

**Remember:** This is a learning project. It's okay if things don't work perfectly the first time. The goal is to understand the concepts, not just get it running!

Good luck with your grokking experiment! 🚀
