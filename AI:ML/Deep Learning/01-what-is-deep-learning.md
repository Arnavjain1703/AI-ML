# 01 — What is Deep Learning?

---

## The Big Picture

Before diving into deep learning, let's understand where it sits in the landscape of artificial intelligence.

```
┌─────────────────────────────────────────────┐
│            Artificial Intelligence           │
│  ┌───────────────────────────────────────┐  │
│  │         Machine Learning              │  │
│  │  ┌─────────────────────────────────┐  │  │
│  │  │       Deep Learning             │  │  │
│  │  │                                 │  │  │
│  │  │  (Neural Networks with          │  │  │
│  │  │   multiple hidden layers)       │  │  │
│  │  └─────────────────────────────────┘  │  │
│  │                                       │  │
│  │  Traditional ML:                      │  │
│  │  • Decision Trees                     │  │
│  │  • SVM                                │  │
│  │  • Random Forest                      │  │
│  │  • Linear/Logistic Regression         │  │
│  │  • KNN, Naive Bayes                   │  │
│  └───────────────────────────────────────┘  │
│                                             │
│  Non-ML AI: Rule-based systems, Expert      │
│  systems, Search algorithms                 │
└─────────────────────────────────────────────┘
```

- **AI** = any system that mimics intelligent behavior
- **Machine Learning** = systems that learn patterns from data (instead of being manually programmed)
- **Deep Learning** = ML using neural networks with many layers

---

## Definition

Deep Learning is a **subset of Machine Learning** that uses artificial neural networks with multiple layers (hence "deep") to learn hierarchical representations of data. Each layer transforms the input into a progressively more abstract representation, enabling the model to capture complex patterns that traditional ML algorithms struggle with.

**Core idea:** Instead of manually engineering features, deep learning *learns* the right features automatically from raw data through multiple layers of nonlinear transformations.

---

## A Simple Analogy

Think of how you recognize a face:

1. Your eyes detect **edges and colors** (low-level)
2. Your brain assembles edges into **shapes** — nose, eyes, mouth (mid-level)
3. You combine shapes into a **face identity** — "That's my friend!" (high-level)

Deep learning does the same thing with data. Each layer learns more abstract features:

```
Image pixels → Edges → Textures → Parts → Objects → "It's a cat!"
```

---

## Why "Deep"?

"Deep" refers to the number of layers. A network with 2-3 layers is "shallow." Modern networks have dozens to hundreds of layers:

| Network | Year | Layers | What it does |
|---------|------|--------|--------------|
| LeNet-5 | 1998 | 7 | Handwritten digit recognition |
| AlexNet | 2012 | 8 | Image classification (ImageNet breakthrough) |
| VGG-16 | 2014 | 16 | Image classification |
| ResNet | 2015 | 152 | Image classification (skip connections) |
| GPT-4 | 2023 | ~120 | Language understanding & generation |

More layers = more abstraction = ability to learn more complex patterns.

---

## Key Takeaway

Deep learning is powerful because it automates feature extraction. In traditional ML, a human engineer decides what features matter. In DL, the network figures it out by itself — given enough data and compute.

---

**Next:** [02 — Neural Network Basics](./02-neural-network-basics.md)
