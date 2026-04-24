# Tredence AI Engineering Internship 2026 (Self-Pruning Neural Network)

A PyTorch implementation of a hybrid CNN + feed-forward neural network that **learns to prune itself during training** via learnable sigmoid gates and L1 sparsity regularisation — achieving **83.21% test accuracy on CIFAR-10** while pruning up to **88.1%** of classifier weights.

---

## Results

| Lambda (λ) | Test Accuracy | Sparsity | Description |
|:---:|:---:|:---:|:---|
| `1e-6` | 83.21% | 15.2% | Mild pruning — most weights kept, good accuracy |
| `5e-6` | 83.16% | 63.2% | Balanced pruning — strong accuracy with high sparsity |
| `2e-5` | 82.36% | 88.1% | ✅ Best trade-off (highest acc×sparsity score) |

### Training Curves
![Training Curves](results/training_curves.png)

### Gate Distributions — Best Model
![Gate Distribution Best](results/best_gate_distribution.png)

> The bimodal distribution (spike at 0 = pruned, cluster near 1 = active) is the hallmark of successful self-pruning via L1 regularisation.

<details>
<summary>View other plots</summary>

**Per-Layer Sparsity**
![Per-Layer Sparsity](results/per_layer_sparsity.png)

</details>

---

## Architecture

```text
  Input (3×32×32)
      │
      ▼
  ┌─────────────────────────────────────────┐
  │  ConvBackbone (VGG-Style 3-block CNN)   │
  │  Block 1: Conv(3→64)   → BN → ReLU×2 → MaxPool  │
  │  Block 2: Conv(64→128) → BN → ReLU×2 → MaxPool  │
  │  Block 3: Conv(128→256)→ BN → ReLU×2 → MaxPool  │
  │  GlobalAvgPool                          │
  └─────────────────────────────────────────┘
      │
      ▼
  256-dim feature vector
      │
      ▼
  ┌─────────────────────────────────────────┐
  │  PrunableLinear(256 → 512)              │  ← learnable gates
  │  BatchNorm1d → ReLU → Dropout(0.4)      │
  ├─────────────────────────────────────────┤
  │  PrunableLinear(512 → 256)              │  ← learnable gates
  │  BatchNorm1d → ReLU → Dropout(0.4)      │
  ├─────────────────────────────────────────┤
  │  PrunableLinear(256 → 10)               │  ← learnable gates
  └─────────────────────────────────────────┘
      │
      ▼
  Softmax (10 classes)
```

---

## Core Idea: PrunableLinear

Every weight in the linear layers has a learnable **gate** parameter:

```python
gate_ij        = sigmoid(gate_scores_ij)   # always in (0, 1)
pruned_weight  = weight_ij  ×  gate_ij     # element-wise
output         = x @ pruned_weight.T  +  bias
```

The **L1 sparsity loss** penalises the sum of all gate values:

```
Total Loss = CrossEntropy(logits, labels) + λ × Σ sigmoid(gate_scores)
```

**Why L1 drives gates to exactly zero (not just small):**
- L1 gradient w.r.t. gate = constant `+1` regardless of magnitude
- Even a gate at `1e-5` gets the same push toward zero
- L2 gradient = `2g` → vanishes as gate shrinks → never truly prunes
- Result: gates **commit** — either stay near 1.0 (useful) or collapse to 0.0 (pruned)

---

## Training Stack

| Technique | Detail |
|---|---|
| **Architecture** | VGG-style 3-block CNN + Gated MLP Head |
| **Regularisation** | Dropout2d(0.1) + Dropout(0.4) |
| **Augmentation** | RandomCrop, RandomHorizontalFlip, ColorJitter, RandomErasing |
| **Loss** | Label-Smoothing Cross-Entropy (ε=0.1) + L1 Sparsity |
| **Optimiser** | AdamW (lr=1e-3 for weights, 0.05 for gates) |
| **LR Schedule** | OneCycleLR (Linear Warmup + Cosine Annealing) |
| **Gate Initialisation**| +2.0 → σ(2) ≈ 0.88 (nearly open) |

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run the experiment
python self_pruning_network.py
```

This will:
- Download CIFAR-10 (first run only)
- Train the self-pruning network with different λ values
- Print a summary table comparing accuracy vs. sparsity
- Save all plots and metrics to the `results/` directory

**Custom Configuration:**
```bash
# Train for more epochs with custom lambdas
python self_pruning_network.py --epochs 30 --lambdas 1e-6 5e-6 2e-5
```

---

## File Structure

```
.
├── self_pruning_network.py   # Main script (model, training, evaluation, plotting)
├── self_pruning_network_notebook (1).ipynb # Comprehensive Jupyter Notebook
├── REPORT.md                 # Analysis report with results and explanation
├── README.md                 # This file
├── requirements.txt          # Python dependencies
└── results/                  # Generated after running (plots, metrics)
    ├── training_curves.png
    ├── per_layer_sparsity.png
    ├── best_gate_distribution.png
    └── results.json
```

---

## Key Design Decisions

**L1 on sigmoid outputs** (not on gate_scores directly): Penalising `sigmoid(gate_scores)` means the penalty is always bounded in `[0, 1]`, making λ interpretable as a fraction of the maximum possible sparsity penalty. The gradient flows cleanly through `sigmoid'` back into `gate_scores`.

**Differential Learning Rates**: The optimizer uses a significantly larger learning rate (0.05) for the `gate_scores` compared to the base weights (1e-3). This allows the network to swiftly adapt the gates and decisively drop connections while steadily optimizing the remaining weights.

**AdamW & OneCycleLR**: Decoupled weight decay gives better generalisation, and the OneCycleLR schedule (linear warmup + cosine annealing) is best-in-class for training CNNs efficiently.

---

*Submitted for Tredence Studio — AI Agents Engineering Team · Internship Case Study 2026*
