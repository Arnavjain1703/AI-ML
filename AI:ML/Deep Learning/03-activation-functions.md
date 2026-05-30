# 03 — Activation Functions

---

## What is an Activation Function?

An activation function is a **mathematical gate** applied to the output of each neuron. It decides **whether and how much** a neuron should "fire" (pass signal forward).

Without activation functions, a neural network — no matter how many layers — would just be a fancy linear regression. Activation functions introduce **nonlinearity**, allowing the network to learn complex patterns like curves, edges, and abstract concepts.

---

## Why Do We Need Them?

```
Without activation (linear):
    Layer 1: z₁ = W₁·x + b₁
    Layer 2: z₂ = W₂·z₁ + b₂ = W₂·(W₁·x + b₁) + b₂ = (W₂W₁)·x + (W₂b₁ + b₂)
    
    → Still just a LINEAR function of x!
    → Multiple layers collapse into one. Depth is useless.

With activation (nonlinear):
    Layer 1: a₁ = σ(W₁·x + b₁)      ← nonlinear transform
    Layer 2: a₂ = σ(W₂·a₁ + b₂)     ← another nonlinear transform
    
    → Can approximate ANY function (Universal Approximation Theorem)
    → Each layer captures different levels of abstraction
```

---

## Where They Sit in a Neuron

```
   x₁ ──w₁──╲
   x₂ ──w₂───╲    ┌─────────┐     ┌──────────────┐
   x₃ ──w₃────▶───│ Σ(wᵢxᵢ) │────▶│ Activation   │────▶ output (a)
   ...        ╱    │  + bias  │     │ f(z)         │
   xₙ ──wₙ──╱     └─────────┘     └──────────────┘
                        z                  a = f(z)

   z = weighted sum (linear part)
   a = activated output (nonlinear part)
```

---

## The Major Activation Functions

### 1. Sigmoid (Logistic)

```
f(z) = 1 / (1 + e⁻ᶻ)
Output range: (0, 1)
```

| Pros | Cons |
|------|------|
| Smooth, differentiable | **Vanishing gradient** — for large |z|, gradient ≈ 0 |
| Output interpretable as probability | Output not zero-centered |
| Good for binary classification output | Computationally expensive |

**Use:** Output layer for binary classification

---

### 2. Tanh (Hyperbolic Tangent)

```
f(z) = (eᶻ - e⁻ᶻ) / (eᶻ + e⁻ᶻ)
Output range: (-1, 1)
```

| Pros | Cons |
|------|------|
| Zero-centered (helps gradients) | Still vanishing gradient for large |z| |
| Stronger gradients than sigmoid | Computationally expensive |

**Use:** Hidden layers in RNNs, when you need negative outputs

---

### 3. ReLU (Rectified Linear Unit) ★ Most Popular

```
f(z) = max(0, z)
Output range: [0, ∞)
```

| Pros | Cons |
|------|------|
| **No vanishing gradient** for z > 0 | **Dying ReLU** — if z < 0, gradient = 0 forever |
| Computationally cheap (just max) | Not zero-centered |
| Sparse activation → efficient | Unbounded output |
| Converges 6x faster than sigmoid | |

**Use:** Default choice for hidden layers in CNNs, MLPs, most architectures

---

### 4. Leaky ReLU

```
f(z) = z        if z > 0
f(z) = 0.01z   if z ≤ 0
Output range: (-∞, ∞)
```

Fixes the dying ReLU problem — always has a small gradient even for negative inputs.

**Use:** When you notice dying neurons with standard ReLU

---

### 5. ELU (Exponential Linear Unit)

```
f(z) = z              if z > 0
f(z) = α(eᶻ - 1)     if z ≤ 0    (α typically = 1.0)
Output range: (-α, ∞)
```

Smooth everywhere (no kink at 0). Outputs can be negative → closer to zero-centered.

---

### 6. Softmax

```
f(zᵢ) = eᶻⁱ / Σⱼ eᶻʲ    (for each class i)
Output: probability distribution over K classes (sums to 1.0)
```

**Example:**
```
Raw logits: z = [2.0, 1.0, 0.5]

e²·⁰ = 7.39,  e¹·⁰ = 2.72,  e⁰·⁵ = 1.65
Sum = 11.76

Softmax = [7.39/11.76, 2.72/11.76, 1.65/11.76]
        = [0.63,       0.23,       0.14]  → sums to 1.0
```

**Use:** Output layer for multi-class classification

---

### 7. Swish / SiLU

```
f(z) = z × sigmoid(z) = z / (1 + e⁻ᶻ)
```

Discovered by Google Brain. Smooth, non-monotonic. Used in EfficientNet.

---

### 8. GELU (Gaussian Error Linear Unit)

```
f(z) = z × Φ(z)    where Φ is the CDF of standard normal
```

Used in **GPT, BERT, and most modern Transformers**. Smooth approximation of ReLU.

---

## Comparison Table

| Function | Range | Zero-Centered | Dying Neurons | Best For |
|----------|-------|:---:|:---:|----------|
| Sigmoid | (0, 1) | ❌ | ❌ | Binary output layer |
| Tanh | (-1, 1) | ✅ | ❌ | RNN hidden layers |
| ReLU | [0, ∞) | ❌ | ✅ | Hidden layers (default) |
| Leaky ReLU | (-∞, ∞) | ❌ | ❌ | When ReLU neurons die |
| ELU | (-α, ∞) | ~✅ | ❌ | Smooth alternative |
| Softmax | (0,1) each | N/A | N/A | Multi-class output |
| GELU | ≈(-0.17, ∞) | ~❌ | ❌ | Transformers (GPT, BERT) |

---

## How to Choose

```
Output layer:
├── Binary classification (yes/no) → Sigmoid
├── Multi-class (pick one of K) → Softmax
└── Regression (any real number) → Linear (no activation) or ReLU

Hidden layers:
├── Default starting point → ReLU
├── Dying neurons observed → Leaky ReLU or ELU
├── Transformer architecture → GELU
└── RNN/LSTM → Tanh (for hidden state)
```

---

## The Vanishing Gradient Problem

This is why activation choice matters:

```
With sigmoid: max(σ'(z)) = 0.25
After 4 layers: 0.25⁴ = 0.0039 → gradient almost zero!
Early layers barely learn.

With ReLU: derivative = 1 for z > 0
After 4 layers: 1×1×1×1 = 1 → gradient preserved!
All layers learn equally fast.
```

This is why ReLU revolutionized deep learning — it enabled training of networks with 50, 100, even 1000+ layers.

---

**Next:** [04 — Cost and Loss Functions](./04-cost-and-loss-functions.md)
