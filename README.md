# The Self-Pruning Neural Network

A PyTorch implementation of a hybrid CNN + feed-forward neural network that **learns to prune itself during training** using an L1 sparsity penalty on learnable gate scores.

---

## Results

Based on the experiments conducted in the Jupyter Notebook, the model achieved the following performance metrics on the CIFAR-10 dataset with different $\lambda$ values for the sparsity penalty:

| $\lambda$ (Lambda) | Test Accuracy (%) | Sparsity Level (%) |
|:------------------:|------------------:|-------------------:|
| `1e-06`            | 79.41             | 0                  |
| `5e-06`            | 78.69             | 0                  |
| `1e-05`            | 78.24             | 0                  |

> **Interpretation of Initial Results**: The table above demonstrates that for very small values of $\lambda$ (e.g., `1e-06` to `1e-05`), the L1 penalty is too weak to force the gate scores below the $0.01$ threshold. The network prioritizes classification accuracy, achieving nearly 80% on CIFAR-10 after only a few epochs, but ignores the pruning objective. Higher $\lambda$ values are required to observe significant sparsity.

### Gate Distributions
![Gate Distribution](result_graph.png)

> A successful self-pruning network exhibits a distinct shift in its gate value distribution, ideally with a spike at exactly $0.0$ indicating fully pruned connections. 

---

## Architecture

```text
  Input (3×32×32)
      │
      ▼
  ┌─────────────────────────────────────────┐
  │  CNN Feature Extractor                  │
  │  Conv2d(3→32)   → BN → ReLU → MaxPool2d │
  │  Conv2d(32→64)  → BN → ReLU → MaxPool2d │
  │  Conv2d(64→128) → BN → ReLU → MaxPool2d │
  │  Flatten                                │
  └─────────────────────────────────────────┘
      │
      ▼
  2048-dim feature vector
      │
      ▼
  ┌─────────────────────────────────────────┐
  │  PrunableLinear(2048 → 512)             │  ← learnable gates
  │  ReLU                                   │
  ├─────────────────────────────────────────┤
  │  PrunableLinear(512 → 128)              │  ← learnable gates
  │  ReLU                                   │
  ├─────────────────────────────────────────┤
  │  PrunableLinear(128 → 10)               │  ← learnable gates
  └─────────────────────────────────────────┘
      │
      ▼
  Softmax (10 classes)
```

---

## Core Idea: PrunableLinear

Every weight in the linear layers has a learnable **gate** parameter:

```python
gate_ij        = torch.sigmoid(gate_scores_ij)   # always in (0, 1)
pruned_weights = weight_ij * gate_ij             # element-wise
output         = F.linear(x, pruned_weights, bias)
```

The **L1 sparsity loss** penalises the sum of all gate values:

```
Total Loss = CrossEntropy(outputs, labels) + λ × Σ |sigmoid(gate_scores)|
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
| **Architecture** | CNN Feature Extractor + Gated MLP Head |
| **Regularisation** | BatchNorm2d + L1 Sparsity on Gates |
| **Augmentation** | ToTensor, Normalize |
| **Loss** | Cross-Entropy + L1 Sparsity |
| **Optimiser** | Adam (lr=0.001) |
| **Gate Initialisation**| Constant +2.0 → σ(2) ≈ 0.88 (mostly active) |

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run the notebook to see the experiment
jupyter notebook self_pruning_nn.ipynb
```

This will:
- Download the CIFAR-10 dataset into `./data` (first run only)
- Train the self-pruning network with $\lambda$ values of `1e-6`, `5e-6`, and `1e-5`
- Print a summary table comparing test accuracy vs. sparsity level
- Visualize the final gate value distribution

---

## File Structure

```
.
├── self_pruning_nn.ipynb     # Jupyter Notebook containing the full implementation and experiment loop
├── self_pruning_report.md    # Detailed technical report analyzing the mechanism and results
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── result_graph.png          # Extracted gate distribution visualization
└── data/                     # Directory for CIFAR-10 dataset
```

---

*Submitted for Tredence Studio — AI Agents Engineering Team · Internship Case Study 2026*
