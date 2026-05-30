# 06 — Optimizers

---

## Gradient Descent vs. Optimizers — What's the Difference?

| | Gradient Descent | Optimizers |
|--|--|--|
| **What it is** | The fundamental algorithm: move weights in the direction that reduces loss | Smarter strategies built on top of gradient descent |
| **Update rule** | `w = w - α × gradient` (one fixed learning rate for everything) | Add momentum, adaptive per-parameter learning rates, etc. |
| **Analogy** | Walking downhill with a fixed step size | Walking downhill with speed-up on slopes, careful steps on rough terrain |

Think of it this way:

- **Gradient Descent** = the basic recipe (compute slope, take a step downhill)
- **Optimizers (Adam, RMSProp, Momentum, etc.)** = enhanced versions of that recipe that solve real-world problems like oscillation, saddle points, and slow convergence

Every optimizer still uses gradients — they just use them more cleverly. All optimizers *are* gradient descent, but plain "vanilla" gradient descent is the simplest optimizer with no enhancements — it's the baseline that others improve upon.

---

## What is an Optimizer?

An optimizer is an algorithm that decides **how to update weights** based on the gradients computed during backpropagation. Plain gradient descent works, but smarter optimizers converge faster and handle tricky loss landscapes better.

---

## The Problem with Basic Gradient Descent

Basic GD uses a single fixed learning rate for all parameters. This causes:
- Slow convergence in flat regions
- Oscillation in steep regions
- Getting stuck in saddle points

Optimizers solve these problems with adaptive learning rates and momentum.

---

## Common Optimizers

### 1. SGD (Stochastic Gradient Descent)

```
w = w - α × gradient
```

The simplest optimizer. Updates weights using the gradient from a random mini-batch.

| Pros | Cons |
|------|------|
| Simple, well-understood | Can be slow, oscillates |
| Good generalization | Sensitive to learning rate choice |

---

### 2. SGD + Momentum

```
v = β × v + gradient           ← accumulate velocity
w = w - α × v                  ← update with velocity
```

**Intuition:** Like a ball rolling downhill — it gains speed in consistent directions and dampens oscillations.

| Pros | Cons |
|------|------|
| Faster than plain SGD | Extra hyperparameter (β, typically 0.9) |
| Reduces oscillation | |
| Can escape shallow local minima | |

---

### 3. AdaGrad (Adaptive Gradient)

```
cache = cache + gradient²
w = w - α × gradient / (√cache + ε)
```

**Key idea:** Per-parameter learning rate. Parameters with frequent large gradients get smaller updates; rare features get larger updates.

| Pros | Cons |
|------|------|
| Good for sparse data (NLP) | Learning rate shrinks monotonically → can stop learning |

---

### 4. RMSProp

```
cache = β × cache + (1 - β) × gradient²
w = w - α × gradient / (√cache + ε)
```

**Key idea:** Fixes AdaGrad's shrinking learning rate by using an exponential moving average instead of accumulating all historical gradients.

| Pros | Cons |
|------|------|
| Doesn't stop learning like AdaGrad | Less popular than Adam |
| Works well for RNNs | |

---

### 5. Adam (Adaptive Moment Estimation) ★ Most Popular

```
m = β₁ × m + (1 - β₁) × gradient          ← first moment (momentum)
v = β₂ × v + (1 - β₂) × gradient²         ← second moment (velocity)

# Bias correction (important for early steps):
m̂ = m / (1 - β₁ᵗ)
v̂ = v / (1 - β₂ᵗ)

w = w - α × m̂ / (√v̂ + ε)                  ← update rule
```

**Key idea:** Combines the best of Momentum (first moment) and RMSProp (second moment). Each parameter gets its own adaptive learning rate PLUS momentum.

Default hyperparameters:
```
α = 0.001, β₁ = 0.9, β₂ = 0.999, ε = 1e-8
```

| Pros | Cons |
|------|------|
| Works out-of-the-box for most problems | Can converge to worse solution than tuned SGD |
| Adaptive per-parameter learning rate | Slightly more memory (stores m and v) |
| Fast convergence | |
| **Industry standard** | |

---

## Comparison Summary

| Optimizer | Key Enhancement | When to Use |
|-----------|----------------|-------------|
| **SGD** | None (baseline) | When you want maximum control |
| **Momentum** | Velocity accumulation | General training, faster than SGD |
| **AdaGrad** | Per-parameter rates | Sparse data, NLP embeddings |
| **RMSProp** | Fixes AdaGrad decay | RNNs, non-stationary problems |
| **Adam** | Momentum + adaptive rates | **Default choice for everything** |

---

## Practical Advice

1. **Start with Adam** — it works well without much tuning
2. If you want the absolute best generalization (e.g., image classification competitions), try **SGD + Momentum** with a learning rate schedule
3. Use **learning rate scheduling** (reduce LR over time) for fine-tuning:
   - Step decay: halve LR every N epochs
   - Cosine annealing: smooth LR reduction
   - Warmup: start small, ramp up, then decay

---

**Next:** [07 — Regularization](./07-regularization.md)
