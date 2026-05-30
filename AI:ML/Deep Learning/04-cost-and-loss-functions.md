# 04 — Cost and Loss Functions

---

## What is "Cost"?

The **cost** (also called **loss** or **error**) is a single number that tells you **how wrong** your neural network's predictions are compared to the actual expected outputs. The goal of training is to minimize this cost.

```
Cost = f(predicted_output, actual_output)
```

- **Low cost** → model predictions are close to actual values → good model
- **High cost** → model predictions are far from actual values → bad model

---

## Cost Function vs. Loss Function

| Term | Scope |
|------|-------|
| **Loss** | Error for a **single** training example |
| **Cost** | Average loss across the **entire** training set (or mini-batch) |

```
Cost J(w, b) = (1/m) × Σ Loss(ŷᵢ, yᵢ)    for i = 1 to m (all training examples)
```

Where:
- `w` = weights, `b` = bias (learnable parameters)
- `ŷᵢ` = predicted output for sample i
- `yᵢ` = actual label for sample i
- `m` = total number of training samples

---

## Common Loss Functions

### 1. Mean Squared Error (MSE)

```
MSE = (1/m) × Σ(ŷ - y)²
```

- **Used for:** Regression (predicting continuous numbers)
- **Intuition:** Penalizes large errors more heavily (squared)
- **Example:** Predicting house prices, temperature

### 2. Binary Cross-Entropy

```
BCE = -(1/m) × Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

- **Used for:** Binary classification (yes/no, spam/not spam)
- **Intuition:** Heavily penalizes confident wrong predictions

### 3. Categorical Cross-Entropy

```
CCE = -(1/m) × Σ Σ yₖ·log(ŷₖ)
```

- **Used for:** Multi-class classification (cat/dog/bird)
- **Intuition:** Only looks at the probability assigned to the correct class

### 4. Mean Absolute Error (MAE)

```
MAE = (1/m) × Σ|ŷ - y|
```

- **Used for:** Regression, more robust to outliers than MSE
- **Intuition:** Treats all errors linearly

### 5. Huber Loss

- MSE when error is small, MAE when error is large
- **Used for:** Regression when you want a balance

---

## Why the Cost Function Matters

The cost function creates a **landscape** (surface) over all possible weight values. Training is about finding the lowest point on this surface — the global minimum:

```
High Cost (bad)          ╲    ╱
                          ╲  ╱
                           ╲╱  ← Global Minimum (lowest cost, best weights)
                     ──────────────────── Weight axis
```

The optimizer (gradient descent) navigates this landscape to find the bottom.

---

## Worked Example: MSE

```
Actual outputs:    y = [1, 0, 1, 1]
Predicted outputs: ŷ = [0.9, 0.1, 0.8, 0.7]

Loss per sample:   (0.9-1)² = 0.01
                   (0.1-0)² = 0.01
                   (0.8-1)² = 0.04
                   (0.7-1)² = 0.09

Cost (MSE) = (0.01 + 0.01 + 0.04 + 0.09) / 4 = 0.0375
```

Training will adjust weights to push this 0.0375 toward 0 (perfect predictions).

---

## Which Loss Function to Use?

| Problem Type | Loss Function | Output Activation |
|-------------|---------------|-------------------|
| Regression | MSE or MAE | Linear (none) |
| Binary classification | Binary Cross-Entropy | Sigmoid |
| Multi-class (one label) | Categorical Cross-Entropy | Softmax |
| Multi-label (multiple labels) | Binary Cross-Entropy per class | Sigmoid per output |

---

**Next:** [05 — Gradient Descent](./05-gradient-descent.md)
