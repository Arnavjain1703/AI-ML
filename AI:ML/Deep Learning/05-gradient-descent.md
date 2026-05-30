# 05 — Gradient Descent

---

## What is Gradient Descent?

Gradient Descent is the **optimization algorithm** that neural networks use to minimize the cost function. It iteratively adjusts weights in the direction that reduces cost the fastest.

**Analogy:** Imagine you're blindfolded on a hilly landscape and need to reach the lowest valley. You feel the slope under your feet and take a step downhill. Repeat until you reach the bottom. That's gradient descent.

---

## The Math

```
w_new = w_old - α × (∂J/∂w)
b_new = b_old - α × (∂J/∂b)
```

Where:
- `w_old` = current weight
- `α` (alpha) = **learning rate** (step size — how big each adjustment is)
- `∂J/∂w` = **gradient** (partial derivative of cost w.r.t. weight) = slope direction
- `w_new` = updated weight

---

## Visual Intuition

```
                Cost J
                  │
          ╲       │       ╱
           ╲      │      ╱
            ╲     │     ╱
             ╲    │    ╱
              ╲   │   ╱
               ╲──┼──╱  ← Minimum (gradient = 0)
                  │
    ──────────────┼──────────── Weight w
                  │
    
    • If slope (gradient) is POSITIVE → w is too large → decrease w
    • If slope (gradient) is NEGATIVE → w is too small → increase w
    • If slope (gradient) is ZERO → w is at minimum → stop
```

The negative sign in the update rule ensures we always move *opposite* to the gradient (downhill).

---

## Learning Rate (α)

The learning rate controls **how big** each step is:

| Learning Rate | Behavior |
|---------------|----------|
| **Too small** (0.00001) | Training is extremely slow, may never converge |
| **Just right** (0.001 - 0.01) | Smooth convergence to minimum |
| **Too large** (1.0+) | Overshoots the minimum, may diverge (cost increases!) |

```
Too small:        Just right:       Too large:
  ╲     ╱          ╲     ╱          ╲     ╱
   ·····            ╲   ╱            ╲   ╱
   (slow)            ╲ ╱             ╳ ╳  (bouncing!)
                      ·               ╳
                   (converged)     (diverging)
```

---

## Types of Gradient Descent

| Variant | Batch Size | Speed | Stability |
|---------|-----------|-------|-----------|
| **Batch GD** | Entire dataset | Slow per step, fewer steps | Most stable, smooth |
| **Stochastic GD (SGD)** | 1 sample | Fast per step, many steps | Noisy, can escape local minima |
| **Mini-Batch GD** | 32-512 samples | Best tradeoff | Moderate noise, GPU-efficient |

### Mini-Batch (Most Common in Practice)

```
For each epoch:
    Shuffle training data
    Split into mini-batches of size B (e.g., 32)
    For each mini-batch:
        1. Forward pass → compute predictions
        2. Compute cost for this mini-batch
        3. Backward pass (backpropagation) → compute gradients
        4. Update weights: w = w - α × gradient
```

---

## Complete Worked Example

```
Given: Input x=2, Actual y=10, Weight w=1, Bias b=0, Learning Rate α=0.1
Loss function: MSE = (ŷ - y)²

Step 1 — Forward Pass:
    ŷ = w×x + b = 1×2 + 0 = 2

Step 2 — Compute Loss:
    L = (2 - 10)² = 64   (very high — bad prediction)

Step 3 — Compute Gradients:
    ∂L/∂w = 2×(ŷ - y)×x = 2×(2-10)×2 = -32
    ∂L/∂b = 2×(ŷ - y)   = 2×(2-10)   = -16

Step 4 — Update Weights:
    w_new = 1 - 0.1×(-32) = 1 + 3.2 = 4.2
    b_new = 0 - 0.1×(-16) = 0 + 1.6 = 1.6

Step 5 — Verify (next forward pass):
    ŷ = 4.2×2 + 1.6 = 10.0   ← Much closer to y=10!
    L = (10.0 - 10)² = 0     ← Perfect!
```

(In practice this takes many iterations, but the concept is the same.)

---

## Common Problems

| Problem | Description | Solution |
|---------|-------------|----------|
| **Local Minima** | Gets stuck in a valley that isn't the lowest | Momentum, Adam optimizer |
| **Saddle Points** | Gradient = 0 but not a minimum | Adam, RMSProp |
| **Vanishing Gradients** | Gradients shrink to ~0 in deep networks | ReLU, batch norm, skip connections |
| **Exploding Gradients** | Gradients grow huge → weights blow up | Gradient clipping, proper initialization |
| **Oscillation** | Zig-zagging instead of going straight | Momentum, learning rate scheduling |

---

**Next:** [06 — Optimizers](./06-optimizers.md)
