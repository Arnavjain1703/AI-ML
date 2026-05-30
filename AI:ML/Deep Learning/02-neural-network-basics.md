# 02 — Neural Network Basics

---

## The Architecture

A neural network is built from layers of connected "neurons":

```
Input Layer → Hidden Layer 1 → Hidden Layer 2 → ... → Hidden Layer N → Output Layer
   (raw data)     (low-level         (mid-level          (high-level      (prediction)
                   features)          features)            features)
```

- **Input layer**: Receives raw data (pixel values, numbers, text embeddings)
- **Hidden layers**: Where the "learning" happens — extract patterns
- **Output layer**: Produces the final prediction (class label, number, probability)

---

## What a Single Neuron Does

Each neuron performs a simple computation:

```
output = activation_function(w₁x₁ + w₂x₂ + ... + wₙxₙ + bias)
```

In plain English:
1. Take inputs (x₁, x₂, ... xₙ)
2. Multiply each by a weight (how important is this input?)
3. Add them all up
4. Add a bias (a constant offset)
5. Pass through an activation function (introduce nonlinearity)
6. Output the result

**Analogy:** A neuron is like a voter. It looks at evidence (inputs), weighs how important each piece is (weights), and makes a decision (output).

---

## Key Components

| Component | What It Does | Analogy |
|-----------|-------------|---------|
| **Neurons** | Compute weighted sum + activation | A decision-making unit |
| **Layers** | Groups of neurons at the same depth | A team of voters |
| **Weights** | Learnable values connecting neurons | "How much do I trust this input?" |
| **Bias** | Offset term — shifts the activation | "My default assumption" |
| **Activation Functions** | Introduce nonlinearity | "Am I excited enough to fire?" |
| **Loss Function** | Measures prediction error | "How wrong was I?" |
| **Optimizer** | Updates weights to reduce error | "How do I improve next time?" |
| **Backpropagation** | Computes gradients for each weight | "Who's responsible for the mistake?" |

---

## The Training Process

Training a neural network is an iterative loop:

```
┌───────────────────────────────────────────────────────────┐
│                    TRAINING LOOP                           │
│                                                           │
│  ┌─────────────┐     ┌────────────┐     ┌─────────────┐ │
│  │ Forward Pass │────▶│ Compute    │────▶│ Backward    │ │
│  │ (predict)    │     │ Cost/Loss  │     │ Pass (grads)│ │
│  └─────────────┘     └────────────┘     └──────┬──────┘ │
│         ▲                                       │        │
│         │              ┌────────────┐           │        │
│         └──────────────│ Update     │◀──────────┘        │
│                        │ Weights    │                     │
│                        │ (GD step)  │                     │
│                        └────────────┘                     │
│                                                           │
│  Repeat until cost converges or max epochs reached        │
└───────────────────────────────────────────────────────────┘
```

### Step by step:

1. **Forward Pass** — Input flows through all layers, producing a prediction
2. **Loss Calculation** — Compare prediction with the actual correct answer
3. **Backward Pass (Backpropagation)** — Compute the gradient of the loss with respect to each weight using the chain rule of calculus
4. **Weight Update** — Optimizer adjusts weights in the direction that reduces loss
5. **Repeat** for many epochs (full passes through the dataset) until the model converges

---

## Terminology You'll See

| Term | Meaning |
|------|---------|
| **Epoch** | One complete pass through the entire training dataset |
| **Batch** | A subset of training examples processed together |
| **Iteration** | One weight update (one batch processed) |
| **Parameters** | Weights + biases (what the network learns) |
| **Hyperparameters** | Settings you choose: learning rate, batch size, number of layers |
| **Inference** | Using a trained model to make predictions on new data |

### Understanding Epoch in Detail

**An epoch = one complete pass through your entire training dataset.**

Imagine you have 1000 training images. Once the network has seen all 1000 images and updated its weights accordingly — that's 1 epoch done.

**Analogy:** Think of studying for an exam. You have a textbook with 10 chapters.
- Reading the entire textbook once = **1 epoch**
- Reading it a second time (you learn more details) = **2 epochs**
- Reading it 50 times = **50 epochs**

Each time through, the model picks up patterns it missed before.

**How epoch, batch, and iteration connect:**

Say you have 1000 samples and a batch size of 100:

```
1 epoch = 1000 / 100 = 10 iterations (10 weight updates)

Epoch 1:  [batch1] → update → [batch2] → update → ... → [batch10] → update  ← done
Epoch 2:  shuffle data, repeat all 10 batches again
...
Epoch 50: by now the model has seen every example 50 times
```

**Why multiple epochs?** One pass isn't enough — the model needs repeated exposure to learn the patterns properly. It's like the difference between reading a textbook once vs. studying it multiple times before an exam.

**How many epochs?** Typically 10–100+, depending on the problem. You stop when validation loss stops improving (early stopping).

---

## A Minimal Example

Imagine a network with 1 input, 1 hidden neuron, and 1 output trying to learn `y = 2x`:

```
Input: x = 3, Expected output: y = 6

Initial state: weight = 0.5, bias = 0

Forward:  ŷ = 0.5 × 3 + 0 = 1.5  (wrong! expected 6)
Loss:     (1.5 - 6)² = 20.25     (high error)
Backward: compute gradient → tells us to increase weight
Update:   weight increases toward 2.0
...after many iterations...
Final:    weight ≈ 2.0, ŷ = 2.0 × 3 = 6 ✓
```

---

**Next:** [03 — Activation Functions](./03-activation-functions.md)
