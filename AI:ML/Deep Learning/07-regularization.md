# 07 — Regularization

---

## What is Regularization?

Regularization is a set of techniques to **prevent overfitting** — when your model memorizes the training data instead of learning general patterns.

```
Underfitting          Good Fit           Overfitting
(too simple)          (just right)       (too complex)

    ·  ·                ·  ·               ·  ·
  ·      ·           ·      ·          ·~~·  ·~~·
 ·        ·         ·    ~~  ·        · ·    ·   ·
──────────────     ───~~──────────    ─~──~──~──~──
(misses pattern)   (captures trend)   (memorizes noise)
```

---

## Common Regularization Techniques

### 1. Dropout

**What:** During training, randomly "turn off" neurons (set their output to 0) with a probability p (typically 0.2–0.5).

**Why it works:** Forces the network to not rely on any single neuron. Creates an implicit ensemble of sub-networks.

```
Training (dropout = 0.5):
  [neuron1] [     ] [neuron3] [     ] [neuron5]  ← random 50% off

Inference (no dropout):
  [neuron1] [neuron2] [neuron3] [neuron4] [neuron5]  ← all active, outputs scaled
```

---

### 2. Batch Normalization

**What:** Normalizes the input to each layer (zero mean, unit variance), then scales and shifts with learnable parameters.

```
x̂ = (x - μ_batch) / √(σ²_batch + ε)
output = γ × x̂ + β    (γ and β are learned)
```

**Why it works:**
- Stabilizes training (reduces internal covariate shift)
- Allows higher learning rates
- Acts as mild regularization
- Faster convergence

---

### 3. L2 Regularization (Weight Decay)

**What:** Add a penalty to the loss function proportional to the square of the weights:

```
Total Loss = Original Loss + λ × Σ(w²)
```

**Why it works:** Discourages large weights → simpler model → less overfitting.

---

### 4. Data Augmentation

**What:** Generate more training data by applying random transformations:

| For Images | For Text | For Audio |
|-----------|----------|-----------|
| Flip, rotate, crop | Synonym replacement | Time stretch |
| Color jitter | Random insertion/deletion | Pitch shift |
| Zoom, translate | Back-translation | Add noise |
| Cutout / MixUp | Sentence shuffling | SpecAugment |

**Why it works:** Exposes the model to more variation → learns robust features.

---

### 5. Early Stopping

**What:** Monitor validation loss during training. Stop when it starts increasing (the model is beginning to overfit).

```
Training Loss:    ↓↓↓↓↓↓↓↓↓↓↓↓  (keeps decreasing)
Validation Loss:  ↓↓↓↓↓↓↑↑↑↑↑↑  (starts increasing here!)
                              ↑
                        STOP HERE
```

---

## When to Use What

| Technique | Best For | Cost |
|-----------|----------|------|
| **Dropout** | Dense layers, FC networks | Slightly longer training |
| **Batch Norm** | CNNs, deep networks | Extra computation per layer |
| **L2 / Weight Decay** | Any model | Almost free |
| **Data Augmentation** | Image/audio tasks | Data pipeline overhead |
| **Early Stopping** | Any model | Need validation set |

In practice, you often combine multiple techniques together.

---

**Next:** [08 — Architectures](./08-architectures.md)
