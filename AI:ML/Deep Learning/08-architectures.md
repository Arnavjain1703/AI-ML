# 08 — Deep Learning Architectures

---

## Overview

Different data types need different network structures. Here are the major architecture families:

---

## 1. Feedforward Neural Networks (FNN / MLP)

**What:** The simplest form — data flows in one direction through dense layers.

```
Input → Dense → Dense → ... → Output
```

- **Data type:** Tabular, structured data
- **Use cases:** Classification, regression on feature vectors
- **Layers:** Fully Connected (Dense)
- **Example:** Predicting house prices from features (size, location, age)

---

## 2. Convolutional Neural Networks (CNN)

**What:** Designed for spatial data. Uses sliding filters to detect local patterns.

```
Image → [Conv → ReLU → Pool] × N → Flatten → Dense → Output
```

**Key idea:** A small filter slides over the image, detecting features like edges, textures, and objects regardless of where they appear.

- **Data type:** Images, video, any grid-structured data
- **Key layers:** Conv2D, MaxPooling, BatchNorm, Flatten
- **Use cases:** Image classification, object detection, facial recognition, medical imaging
- **Famous models:** LeNet, AlexNet, VGG, ResNet, EfficientNet

---

## 3. Recurrent Neural Networks (RNN)

**What:** Designed for sequential data. Has memory — each step's output depends on previous steps.

```
x₁ → [h₁] → x₂ → [h₂] → x₃ → [h₃] → ... → output
       ↑              ↑              ↑
    (memory)      (memory)       (memory)
```

**Problem:** Basic RNNs forget long-range dependencies (vanishing gradient over time).

### Variants:

| Variant | Key Innovation |
|---------|---------------|
| **LSTM** (Long Short-Term Memory) | Forget gate + cell state → remembers long sequences |
| **GRU** (Gated Recurrent Unit) | Simplified LSTM — faster, similar performance |

- **Data type:** Text, time series, speech, any sequential data
- **Use cases:** Language modeling, translation, speech recognition
- **Mostly replaced by:** Transformers (for NLP)

---

## 4. Transformers ★ Dominant Architecture

**What:** Attention-based architecture. Processes entire sequences in parallel (no recurrence).

**Key innovation:** **Self-Attention** — each word/token can "look at" every other word to understand context.

```
Input → [Multi-Head Attention → Feed Forward] × N → Output
```

- **Data type:** Text, code, images (ViT), audio, video
- **Use cases:** LLMs (GPT, Claude), translation (BERT), code generation, image generation
- **Why they won:** Parallelizable (fast training on GPUs), captures long-range dependencies perfectly

| Model | Type | What It Does |
|-------|------|-------------|
| BERT | Encoder | Understanding, classification |
| GPT | Decoder | Text generation |
| T5 | Encoder-Decoder | Translation, summarization |
| ViT | Encoder (images) | Image classification |

---

## 5. Generative Adversarial Networks (GANs)

**What:** Two networks compete against each other:

```
Generator (G):  Random noise → Fake image
Discriminator (D): Image → "Real" or "Fake"

G tries to fool D. D tries to catch G. Both improve.
```

- **Use cases:** Image generation, style transfer, super-resolution, deepfakes
- **Famous models:** StyleGAN, CycleGAN, Pix2Pix

---

## 6. Autoencoders

**What:** Learn compressed representations by encoding input to a small bottleneck, then decoding back.

```
Input → [Encoder] → Bottleneck (small) → [Decoder] → Reconstruction
```

- **Use cases:** Anomaly detection, denoising, dimensionality reduction, data compression
- **Variant:** VAE (Variational Autoencoder) — generates new data samples

---

## Architecture Selection Guide

```
What's your data?
├── Images/Video → CNN (or ViT for large-scale)
├── Text/Language → Transformer
├── Time series → LSTM/GRU (or Transformer)
├── Tabular data → MLP (or traditional ML — often better!)
├── Generate new content → GAN or Diffusion Model
└── Compress / detect anomalies → Autoencoder
```

---

**Next:** [09 — Deep Learning vs ML](./09-deep-learning-vs-ml.md)
