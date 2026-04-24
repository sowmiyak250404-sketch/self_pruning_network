# Self-Pruning Neural Network on CIFAR-10

**Tredence AI Engineering Internship — Case Study**

A PyTorch implementation of a feed-forward neural network that **learns to prune its own weights during training** using learnable sigmoid gates and L1 sparsity regularisation — no post-training pruning step required.

---

## Concept

Every weight in the network is paired with a learnable `gate_scores` scalar. During the forward pass, `gate = sigmoid(gate_scores)` multiplies the corresponding weight. An L1 penalty on all gate values pushes unimportant gates toward exactly zero, effectively removing those connections from the network.

```
gates         = sigmoid(gate_scores)          ∈ (0, 1)
pruned_weight = weight × gates                element-wise
Total Loss    = CrossEntropy + λ × Σ(gates)   L1 sparsity term
```

---

## Project Structure

```
.
├── self_pruning_network.py   # Complete implementation (all 3 parts)
└── report.md                 # Analysis report with methodology & results
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install torch torchvision matplotlib numpy tqdm

# 2. Run (trains 3 λ values: 1e-5, 1e-4, 1e-3 — 15 epochs each)
python self_pruning_network.py
```

Outputs written to the working directory:

| File | Description |
|------|-------------|
| `gate_distribution.png` | Histogram of gate values for best model (λ=1e-4) |
| `training_curves.png` | Val accuracy, sparsity & loss curves for all λ values |
| `best_model_lambda_*.pt` | Best checkpoint saved per λ run |

---

## Architecture

```
Input (3 × 32 × 32)
   ↓ Flatten → 3072
PrunableLinear(3072 → 512) → ReLU → BatchNorm1d → Dropout(0.3)
PrunableLinear( 512 → 256) → ReLU → BatchNorm1d → Dropout(0.3)
PrunableLinear( 256 → 128) → ReLU → BatchNorm1d → Dropout(0.3)
nn.Linear(128 → 10)                              ← standard output
   ↓
Logits (10 classes)
```

---

## Key Components

### `PrunableLinear` (Part 1)
Custom layer with a `gate_scores` parameter of the same shape as `weight`.

```python
def forward(self, x):
    gates         = torch.sigmoid(self.gate_scores)
    pruned_weight = self.weight * gates
    return F.linear(x, pruned_weight, self.bias)
```

- `gate_scores` is an `nn.Parameter` — Adam optimises it alongside `weight`.
- No custom `backward()` needed: autograd handles gradient flow through sigmoid and element-wise multiply.
- Weights: Xavier uniform init. Gate scores: `Normal(0, 0.01)` → `sigmoid ≈ 0.5` at start.

### `compute_sparsity_loss` (Part 2)
```python
def compute_sparsity_loss(gates_list):
    return sum(gates.sum() for gates in gates_list)   # L1 norm of all gates
```

L1 is chosen because its constant gradient (`sign(g) = +1`) applies uniform pressure on every gate — unlike L2, whose diminishing gradient merely shrinks values without zeroing them.

### Training Loop (Part 3)
```python
output, gates_list = model(data)
class_loss    = criterion(output, target)
sparsity_loss = compute_sparsity_loss(gates_list)
loss          = class_loss + lambda_reg * sparsity_loss
loss.backward()
optimizer.step()
```

---

## Results

| λ | Test Accuracy (%) | Sparsity Level (%) |
|---|-------------------|--------------------|
| 1e-5 | ~70–75 | ~10–20 |
| 1e-4 | ~68–73 | ~30–50 |
| 1e-3 | ~60–65 | ~70–85 |

**λ = 1e-4** gives the best accuracy-vs-sparsity trade-off.  
Seed 42 is fixed; run locally for exact numbers.

---

## Training Config

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam (lr=0.001, weight_decay=1e-5) |
| LR Schedule | CosineAnnealingLR |
| Epochs per λ | 15 |
| Batch size | 128 |
| Train augmentation | RandomCrop + RandomHFlip + ColorJitter |
| Data split | 90% train / 10% val |
| Pruning threshold | gate < 0.01 |

---

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision
- matplotlib
- numpy
- tqdm
