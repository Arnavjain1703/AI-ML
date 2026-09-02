# Supervised Learning — Naive Bayes & K-Fold Cross-Validation

Study notes from the lecture (Prof. Durga Toshniwal). Covers every topic discussed, each with a Mermaid diagram.

---

## 0. Where this lecture sits

```mermaid
mindmap
  root((Supervised Learning))
    Eager Learners
      Decision Tree
    Lazy Learners
      Instance-based
        K-Nearest Neighbor
        Rote Learner
      Naive Bayes
    Model Validation
      Knee-of-curve split
      K-Fold Cross-Validation
    Next topic
      Regression
```

---

## 1. Recap — Instance-Based Classifiers

An instance-based classifier builds **no model**. It just stores all training records and, when an unseen sample arrives, searches the stored data for a match.

```mermaid
flowchart TD
    A[Unseen record arrives] --> B[Store all training records<br/>no model built]
    B --> C{Exact match on<br/>attribute values?}
    C -->|Yes| D[Use that record's class label]
    C -->|No match| E[Classifier fails]
    C -->|Scan one by one| F[High computation<br/>e.g. 10,000 records]
```

**Limitations**
- Needs an **exact match** → often none exists → classifier fails.
- Must scan every training record → very expensive on large data.

**Fix:** allow an **approximate match** → leads to **K-Nearest Neighbor (KNN)**.

---

## 2. K-Nearest Neighbor (KNN)

Each training sample is a point in a **D-dimensional space** (D = number of attributes). For an unseen record, find the **K closest points** using a distance metric (e.g. Euclidean), then take a **majority vote** of their class labels.

```mermaid
flowchart LR
    A[Unseen record] --> B[Map all samples to<br/>D-dimensional space]
    B --> C[Compute distance<br/>Euclidean etc.]
    C --> D[Pick K nearest neighbors]
    D --> E{Class labels<br/>of K points}
    E --> F[Majority vote → class label]
```

### Choosing K (bias–variance tradeoff)

```mermaid
flowchart TD
    A[Choice of K] --> B[K too small e.g. K=1]
    A --> C[K optimal<br/>best performance on graph]
    A --> D[K too large]
    B --> B1[Picks noise points<br/>OVERFITTING]
    C --> C1[Good generalization]
    D --> D1[True class not captured<br/>too generic → UNDERFITTING]
```

Plot **performance vs K** and pick the K where performance is best.

---

## 3. Eager vs Lazy Learners

```mermaid
flowchart LR
    subgraph Eager [Eager Learner e.g. Decision Tree]
        E1[Training time: HIGH<br/>builds a model] --> E2[Prediction time: LOW]
    end
    subgraph Lazy [Lazy Learner e.g. KNN, Naive Bayes]
        L1[Training time: LOW<br/>no model built] --> L2[Prediction time: HIGH<br/>work done at query time]
    end
```

Choice depends on the application (fast training vs fast prediction).

---

## 4. Why we need probabilistic classifiers

Sometimes the **same attribute values map to different class labels** — the relationship is **non-deterministic**.

**Example:** a person on a strict diet + frequent workout — do they risk heart disease?
Not guaranteed "no" — hidden factors (smoking, heredity, stress, sudden events) may not be in the data.

```mermaid
flowchart TD
    A[Same attribute values<br/>diet=good, workout=frequent] --> B[Person 1: NO heart disease]
    A --> C[Person 2: YES heart disease]
    C --> D[Hidden / implicit factors<br/>smoking, stress, heredity]
    A --> E[Relationship is<br/>NON-DETERMINISTIC]
    E --> F[Need a PROBABILISTIC model<br/>→ Bayesian classifiers]
```

---

## 5. Bayes' Theorem

Relates **conditional probability** and **marginal (prior) probability**.

$$P(C \mid A) = \frac{P(A \mid C)\, P(C)}{P(A)}$$

```mermaid
flowchart LR
    subgraph Terms
        T1["P(C | A)<br/>Posterior<br/>C given A occurred"]
        T2["P(A | C)<br/>Likelihood"]
        T3["P(C)<br/>Prior / marginal of C"]
        T4["P(A)<br/>Prior / marginal of A"]
    end
    T2 --> R["P(C|A) = P(A|C)·P(C) / P(A)"]
    T3 --> R
    T4 --> R
```

**Key idea:** `P(C)` (independent) ≠ `P(C | A)` (after observing A).
> Observations change our beliefs — e.g. trust in a teammate after seeing past performance; reliance on an airline after seeing disruptions. Independent events (coin toss, draw-with-replacement) are the exception where they stay equal.

### Worked example — Meningitis & stiff neck

Given: `P(stiff neck | meningitis) = 0.5`, `P(meningitis) = 1/50000`, `P(stiff neck) = 1/20`.

```mermaid
flowchart TD
    A["P(M | S) = P(S | M) · P(M) / P(S)"] --> B["= 0.5 × (1/50000) / (1/20)"]
    B --> C["= 0.0002 → very low"]
    C --> D[Predicted class: NOT meningitis]
```

**Use in ML:** for multiple classes, compute `P(C₁|A), P(C₂|A), …` and pick the **highest** as the predicted label.

---

## 6. Naive Bayes Classifier

Goal: `P(C | A₁, A₂, …, Aₙ)`. Directly estimating `P(A₁…Aₙ | C)` is hard, so Naive Bayes assumes **conditional independence** of attributes given the class.

$$P(A_1,\dots,A_n \mid C_j) = \prod_{i=1}^{n} P(A_i \mid C_j)$$

```mermaid
flowchart TD
    A["P(C | A1,A2,...,An)"] --> B["∝ P(A1,...,An | C) · P(C)"]
    B --> C[Naive assumption:<br/>attributes conditionally<br/>independent given C]
    C --> D["= P(A1|C)·P(A2|C)·...·P(An|C) · P(C)"]
    D --> E[Denominator P(A1..An)<br/>ignored — same for all classes]
    E --> F[Pick class C with<br/>highest product]
```

---

## 7. Naive Bayes — Worked Example (Cheating dataset)

10 training samples, class label = **Cheat (Yes/No)**. Priors: `P(No)=7/10`, `P(Yes)=3/10`.
Continuous **income** is **discretized**: `≥80K` vs `<80K` (70, 60, 75 all fall in `<80K`).

**Query:** `Refund=No, Marital=Single, Income=70K` → predict class.

```mermaid
flowchart TD
    Q[Unseen: Refund=No,<br/>Single, Income=70K] --> N[Class = NO]
    Q --> Y[Class = YES]

    N --> N1["P(Refund=No | No) = 4/7"]
    N --> N2["P(Single | No) = 3/7"]
    N --> N3["P(Income=70K | No) = 3/7"]
    N --> N4["P(No) = 7/10"]
    N1 & N2 & N3 & N4 --> NR["Product = (4/7)(3/7)(3/7)(7/10)<br/>= 36/490 ≈ 0.073"]

    Y --> Y1["P(Refund=No | Yes) = 3/3 = 1"]
    Y --> Y2["P(Single | Yes) = 3/3 = 1"]
    Y --> Y3["P(Income=70K | Yes) = 0/3 = 0"]
    Y --> Y4["P(Yes) = 3/10"]
    Y1 & Y2 & Y3 & Y4 --> YR["Product = 1·1·0·(3/10) = 0"]

    NR --> DEC{Compare}
    YR --> DEC
    DEC --> OUT["P(No) > P(Yes) → Predict NO"]
```

**Key point (attribute independence):** each attribute is counted **independently** — we don't require rows to match on all three attributes at once, only per-attribute given the class.

---

## 8. Properties of Naive Bayes

```mermaid
flowchart TD
    A[Naive Bayes] --> B[Robust to NOISE<br/>noise points have low probability impact]
    A --> C[Handles MISSING VALUES<br/>can predict missing attribute/label]
    A --> D[Robust to IRRELEVANT attributes<br/>~equal probability across classes → no effect]
    A --> E[Assumes attribute INDEPENDENCE]
    E --> F{Attributes dependent?}
    F -->|Yes e.g. time-series| G[Do NOT use Naive Bayes]
    F -->|No e.g. word 'Amazon'<br/>ecommerce vs river| H[Naive Bayes works well]
```

**"Amazon" example:** context decides class — "purchased on Amazon" → e-commerce; "Amazon River is long" → river/rainforest.
**Time-series counter-example:** value at Tₙ depends on previous values → not independent → don't use Naive Bayes.

---

## 9. Naive Bayes vs Decision Tree — when to use which

```mermaid
flowchart TD
    A[Attribute-value combinations] --> B{Unique to<br/>one class label?}
    B -->|Unique| C[Decision Tree / Random Forest OK]
    B -->|NOT unique<br/>same values → different labels| D[Decision tree cannot be drawn]
    D --> E[Use Naive Bayes:<br/>compare P No vs P Yes]
```

Decision trees fail when identical attribute values carry different labels; probabilities resolve it.

---

## 10. K-Fold Cross-Validation

Split training data into **K equal folds**. In each of K iterations, one fold = **test**, the rest = **train**. Average the K performances → final performance (removes data bias).

```mermaid
flowchart TD
    A[Training data] --> B[Split into K equal folds<br/>e.g. K=5]
    B --> I1[Iter 1: fold1=Test, rest=Train → P1]
    B --> I2[Iter 2: fold2=Test, rest=Train → P2]
    B --> I3[Iter 3: fold3=Test → P3]
    B --> I4[Iter 4: fold4=Test → P4]
    B --> I5[Iter 5: fold5=Test → P5]
    I1 & I2 & I3 & I4 & I5 --> AVG["Final performance = avg(P1..P5)<br/>accuracy / precision / recall / F1"]
```

### Why average?
Any single fold may be biased (e.g. a majority class in one fold gives 95%, a hard fold gives 80%). Averaging gives an **unbiased** estimate.

### Choosing the number of folds K

```mermaid
flowchart LR
    A[Very low K e.g. 0<br/>all data train & test] --> A1[Overly generic]
    B[Very high K = N<br/>one point per fold] --> B1[Overfitting +<br/>too many computations]
    C[Optimal K] --> C1[Performance stable,<br/>reasonable compute]
```

Plot **performance vs K**, pick where the curve **flattens** (adding folds beyond adds compute, not accuracy). K-fold and the "training-split ratio / knee-of-curve" method both ultimately decide the **train-test ratio** — the two most popular approaches.

### Two methods to split train/test

```mermaid
flowchart TD
    A[Single labeled dataset] --> B[Method 1: Split-ratio + knee of curve]
    A --> C[Method 2: K-Fold cross-validation]
    B --> B1[Try 10/20/.../90% train,<br/>plot performance, pick knee]
    C --> C1[Auto folds from K,<br/>iterate, average performance]
    B1 --> D[Both decide<br/>train : test ratio]
    C1 --> D
```

---

## Quick reference

| Concept | One-liner |
|---|---|
| Lazy learner | No model; work done at prediction time (KNN, Naive Bayes) |
| Eager learner | Builds model up front (Decision Tree) |
| Bayes theorem | `P(C\|A) = P(A\|C)P(C)/P(A)` |
| Prior / marginal | `P(C)` independent of other events |
| Posterior | `P(C\|A)` after observing A |
| Naive assumption | Attributes conditionally independent given class |
| Ignore denominator | `P(A₁…Aₙ)` constant across classes |
| Use Naive Bayes when | Noise, missing values, non-unique value→label, independent attributes |
| Avoid Naive Bayes when | Attributes dependent (e.g. time series) |
| K-fold CV | K folds, rotate test fold, average performance |
