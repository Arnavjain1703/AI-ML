# 09 — Deep Learning vs. Traditional Machine Learning

---

## Key Differences

| Aspect | Machine Learning | Deep Learning |
|--------|-----------------|---------------|
| **Feature Engineering** | Manual — domain expert selects/creates features | Automatic — network learns features from raw data |
| **Data Requirements** | Works with small-to-medium datasets (100s to 10Ks) | Needs large datasets (10Ks to millions) |
| **Compute Requirements** | Runs on CPU, moderate resources | Needs GPUs/TPUs, high memory |
| **Model Complexity** | Simple to moderate (linear models, trees, SVMs) | Very complex (millions to billions of parameters) |
| **Interpretability** | Often interpretable (decision trees, linear regression) | Black box — hard to explain decisions |
| **Training Time** | Minutes to hours | Hours to weeks |
| **Performance on Structured Data** | Often better (tabular, well-defined features) | Similar or slightly worse |
| **Performance on Unstructured Data** | Struggles (images, audio, text) | Excels — state-of-the-art |
| **Overfitting Risk** | Moderate | High — needs dropout, batch norm, augmentation |
| **Example Algorithms** | Random Forest, SVM, XGBoost, Logistic Regression | CNN, RNN, Transformer, GAN |

---

## When to Use Traditional ML

- Dataset is small (< 10K samples)
- Features are well-defined and structured (tabular data)
- Interpretability is critical (healthcare, finance, legal)
- Compute budget is limited
- Quick prototyping needed

**Examples:** Fraud detection on transaction features, churn prediction, credit scoring

---

## When to Use Deep Learning

- Data is unstructured (images, audio, video, text)
- Dataset is very large (100K+ samples)
- You want automatic feature extraction
- State-of-the-art accuracy is required
- Compute resources (GPU) are available

**Examples:** Self-driving cars, ChatGPT, Google Translate, face unlock, voice assistants

---

## The Hybrid Approach

In production, the best solution often combines both:
- **ML** for business rules, structured data, quick decisions
- **DL** for perception tasks (vision, language, speech)
- **Transfer learning** bridges the gap — use pretrained DL models even with small datasets

---

## Quick Decision Chart

```
Is your data structured/tabular?
├── YES → Use Traditional ML (XGBoost, Random Forest, LightGBM)
└── NO → Is data unstructured (image/text/audio)?
    ├── YES → Do you have >10K labeled samples + GPU?
    │   ├── YES → Use Deep Learning
    │   └── NO → Use pretrained models (transfer learning)
    └── NO → Consider the problem type and data size
```

---

**Next:** [10 — Real-World Applications](./10-real-world-applications.md)
