# Data Similarity, Distance Measures & Evaluation Metrics
> ML lecture notes — data types, similarity measures, confusion matrix, and model evaluation.

---

## Table of Contents
1. [Types of Data Attributes](#1-types-of-data-attributes)
2. [Properties of a Valid Metric](#2-properties-of-a-valid-distance-metric)
3. [Distance Measures for Numeric Data](#3-distance-measures-for-numeric-data)
4. [Distance Matrix](#4-distance-matrix)
5. [Binary Vector Similarity](#5-binary-vector-similarity)
6. [Cosine Similarity](#6-cosine-similarity)
7. [Confusion Matrix](#7-confusion-matrix)
8. [Precision](#81-precision)
9. [Recall](#82-recall)
10. [Precision–Recall Trade-off](#83-precisionrecall-trade-off)
11. [F1 Score](#84-f1-score)
12. [Accuracy](#85-accuracy)
13. [Sensitivity & Specificity](#86-sensitivity--specificity)
14. [FPR & FNR](#87-fpr--fnr)
15. [ROC Curve & AUC](#9-roc-curve--auc)
16. [Cheat Sheet & Decision Guide](#10-complete-cheat-sheet)

---

## 1. Types of Data Attributes

### Taxonomy of Attribute Types

```mermaid
mindmap
  root((Data Attributes))
    Numeric
      Discrete
        Finite values
        e.g. Digits 0-9
        e.g. Count of students
      Continuous
        Infinite values in range
        e.g. Height
        e.g. Weight
        e.g. Temperature
    Categorical
      Binary
        0 or 1
        e.g. Gender
        e.g. Yes/No flags
      Nominal
        No order
        e.g. ZIP codes
        e.g. PIN codes
        e.g. Colors
      Ordinal
        Has order
        e.g. Ratings 1-5
        e.g. Low/Medium/High
```

### Summary Table

| Type | Behaves like a number? | Examples |
|------|----------------------|---------|
| **Numeric** | Yes | Temperature, Height, Weight |
| **Categorical** | No (even if stored as number) | Gender (0/1), ZIP codes, PIN codes |
| **Discrete** | Finite set of values | Digits 0–9, number of students |
| **Continuous** | Infinite values in a range | Weight (70.1, 70.11, 70.111…) |

> **Key insight:** PIN codes look like numbers but represent *locations*. Adding two PIN codes is meaningless — they are categorical, not numeric.

---

## 2. Properties of a Valid Distance Metric

All four properties must hold for `D(P, Q)` to be a valid metric.

```mermaid
graph TD
    M([Valid Distance Metric]) --> S[1. Symmetry\nD P,Q = D Q,P]
    M --> SS[2. Self-Similarity\nD P,P = 0]
    M --> P[3. Positivity\nD P,Q >= 0]
    M --> T[4. Triangle Inequality\nD B,C <= D B,A + D A,C]

    S --> S1["If Alex looks like Bob\n→ Bob looks like Alex"]
    SS --> SS1["Alex looks most like himself\nnot like someone else"]
    P --> P1["Distance is never negative\n= 0 only when P equals Q"]
    T --> T1["Alex resembles Bob AND Carl\n→ Bob and Carl resemble each other"]

    style M fill:#4A90D9,color:#fff
    style S fill:#27AE60,color:#fff
    style SS fill:#27AE60,color:#fff
    style P fill:#27AE60,color:#fff
    style T fill:#27AE60,color:#fff
```

---

## 3. Distance Measures for Numeric Data

### Minkowski Family

```mermaid
graph LR
    MK["Minkowski Distance\nd = ( Σ |Pk-Qk|^r )^(1/r)"]
    MK -->|r = 1| L1["L1 Norm\nManhattan Distance\nΣ |Pk - Qk|"]
    MK -->|r = 2| L2["L2 Norm\nEuclidean Distance\n√( Σ (Pk-Qk)² )"]
    MK -->|r = ∞| Linf["L∞ Norm\nChebyshev Distance\nmax |Pk - Qk|"]

    L1 --> L1ex["Example P=(0,2) Q=(2,0)\n|0-2| + |2-0| = 4"]
    L2 --> L2ex["Example P=(0,2) Q=(2,0)\n√(4+4) = 2.83"]

    style MK fill:#8E44AD,color:#fff
    style L1 fill:#2980B9,color:#fff
    style L2 fill:#16A085,color:#fff
    style Linf fill:#D35400,color:#fff
```

### Manhattan vs Euclidean — Concept

```mermaid
graph LR
    A["Point A\n(0, 0)"] -->|"Euclidean\n(diagonal, √8 ≈ 2.83)"| B["Point B\n(2, 2)"]
    A -->|"Manhattan\n(grid, 2+2 = 4)"| R1["→ (2,0)"]
    R1 -->|"then up"| B

    style A fill:#2ECC71,color:#fff
    style B fill:#E74C3C,color:#fff
    style R1 fill:#F39C12,color:#fff
```

> Euclidean = straight line (shortest).  Manhattan = city-block path (horizontal + vertical only).

---

## 4. Distance Matrix

### How it is built

```mermaid
flowchart TD
    D["N data points\n(each with attributes)"] --> PM["Compute pairwise\ndistances using\nL1 or L2 norm"]
    PM --> DM["N × N Distance Matrix"]
    DM --> D0["Diagonal = 0\n(self-similarity)"]
    DM --> DS["Symmetric\nD(Pi,Pj) = D(Pj,Pi)"]
    DM --> DU["Only upper or lower\ntriangle needed"]
    DM --> USE["Used for:\nClustering / Grouping\nNearest Neighbor Search"]

    style D fill:#2980B9,color:#fff
    style DM fill:#8E44AD,color:#fff
    style USE fill:#27AE60,color:#fff
```

### Worked Example — 4 points in 2D

| Point | X | Y |
|-------|---|---|
| P1 | 0 | 2 |
| P2 | 2 | 0 |
| P3 | 3 | 1 |
| P4 | 5 | 1 |

#### Euclidean (L2) Distance Matrix

|    | P1 | P2 | P3 | P4 |
|----|----|----|----|----|
| **P1** | 0 | 2.83 | 3.16 | 5.10 |
| **P2** | 2.83 | 0 | **1.41** | 3.00 |
| **P3** | 3.16 | 1.41 | 0 | 2.00 |
| **P4** | 5.10 | 3.00 | 2.00 | 0 |

> **Closest pair: P2 and P3** (d = √2 ≈ 1.414)

---

## 5. Binary Vector Similarity

### When to use SMC vs Jaccard

```mermaid
flowchart TD
    Q{"Does absence (0-0 match)\ncarry meaningful information?"}
    Q -->|Yes| SMC["Use Simple Matching\nCoefficient (SMC)\n\nSMC = (M11+M00) / Total"]
    Q -->|No| JAC["Use Jaccard Coefficient\n\nJ = M11 / (M11+M10+M01)"]

    SMC --> EX1["Example:\nDistrict gender majority\nBoth male-majority districts\nare genuinely similar"]
    JAC --> EX2["Example:\nShopping baskets\nTwo people who both\ndid NOT buy bread\nare NOT similar for that reason"]

    style Q fill:#F39C12,color:#fff
    style SMC fill:#2980B9,color:#fff
    style JAC fill:#27AE60,color:#fff
```

### Match Type Breakdown

```mermaid
quadrantChart
    title Binary Vector Match Types
    x-axis "Vector Q value" 0 --> 1
    y-axis "Vector P value" 0 --> 1
    quadrant-1 M11 - Both present (counts in SMC and Jaccard)
    quadrant-2 M10 - P present Q absent (mismatch)
    quadrant-3 M00 - Both absent (counts in SMC only)
    quadrant-4 M01 - P absent Q present (mismatch)
```

### Worked Example

```
P: 1  0  0  1  0  0  0  0  0  0
Q: 0  0  1  1  0  0  0  0  0  0
         ↑  ↑
        M01 M11  ← only 1 shared purchase
```

| Count | Value |
|-------|-------|
| M₁₁ | 1 |
| M₀₀ | 5 |
| M₁₀ | 2 |
| M₀₁ | 2 |

| Metric | Calculation | Score | Why it differs |
|--------|------------|-------|----------------|
| **SMC** | (1+5)/(1+5+2+2) | **60%** | 5 "neither bought" matches inflate result |
| **Jaccard** | 1/(1+2+2) | **20%** | Only real shared purchases counted |

---

## 6. Cosine Similarity

```mermaid
graph LR
    D1["Document D1\n(word frequency vector)"] --> COS
    D2["Document D2\n(word frequency vector)"] --> COS
    COS["cos θ = (D1·D2) / (|D1|×|D2|)"] --> RANGE["Range: 0 to 1\n0 = completely different\n1 = identical direction"]
    RANGE --> USE["Most popular for\nDocument Similarity\nText Search\nRecommendation Systems"]

    style COS fill:#8E44AD,color:#fff
    style RANGE fill:#16A085,color:#fff
```

### Worked Example

|    | team | coach | play | ball | score |
|----|------|-------|------|------|-------|
| D1 | 3 | 2 | 2 | 5 | 1 |
| D2 | 1 | 0 | 2 | 3 | 2 |

```
D1·D2 = (3×1)+(2×0)+(2×2)+(5×3)+(1×2) = 24
|D1|  = √(9+4+4+25+1) = √43 ≈ 6.56
|D2|  = √(1+0+4+9+4)  = √18 ≈ 4.24

cos(θ) = 24 / (6.56 × 4.24) ≈ 0.86  → High similarity
```

---

## 7. Confusion Matrix

```mermaid
graph TD
    subgraph MATRIX["Confusion Matrix"]
        TP["✅ True Positive (TP)\nActually +ve\nPredicted +ve"]
        FN["❌ False Negative (FN)\nActually +ve\nPredicted -ve\n(MISS)"]
        FP["❌ False Positive (FP)\nActually -ve\nPredicted +ve\n(FALSE ALARM)"]
        TN["✅ True Negative (TN)\nActually -ve\nPredicted -ve"]
    end

    TP --- FN
    FP --- TN

    style TP fill:#27AE60,color:#fff
    style TN fill:#27AE60,color:#fff
    style FP fill:#E74C3C,color:#fff
    style FN fill:#E74C3C,color:#fff
```

### COVID Test Example

| | Predicted: Positive | Predicted: Negative |
|-|---------------------|---------------------|
| **Actual: Positive (sick)** | TP ✅ Correctly flagged | FN ❌ Sick walks free — dangerous |
| **Actual: Negative (healthy)** | FP ❌ Healthy quarantined | TN ✅ Correctly cleared |

---

## 8.1 Precision

```
Precision = TP / (TP + FP)
```

```mermaid
pie title Predicted Positives breakdown
    "True Positive (correctly flagged)" : 70
    "False Positive (false alarm)" : 30
```

> Precision = 70 / (70+30) = **70%** — 30% of positive predictions were wrong.

- High precision = **low FP** = model rarely raises false alarms
- Model is **conservative**

**Real-world stakes:**

| Scenario | FP means | Impact |
|----------|---------|--------|
| Spam filter | Legit email blocked | User misses important email |
| Legal conviction | Innocent jailed | Severe injustice |
| Medical test | Healthy person treated | Unnecessary trauma/cost |

---

## 8.2 Recall

```
Recall = TP / (TP + FN)
```

```mermaid
pie title Actual Positives breakdown
    "True Positive (correctly caught)" : 60
    "False Negative (missed)" : 40
```

> Recall = 60 / (60+40) = **60%** — 40% of real positives were missed.

- High recall = **low FN** = model misses very few real positives
- Model is **lenient**

**Real-world stakes:**

| Scenario | FN means | Impact |
|----------|---------|--------|
| COVID/Cancer test | Sick person declared healthy | Spreads disease / untreated cancer |
| Fraud detection | Real fraud approved | Financial loss |
| Factory fault | Real fault undetected | Breakdown, safety incident |

---

## 8.3 Precision–Recall Trade-off

```mermaid
graph LR
    subgraph STRICT["High Precision (Strict Model)"]
        direction TB
        A1["Rarely predicts positive\nFP stays low ✅"] --> A2["Misses real positives\nFN rises ❌"] --> A3["Recall goes DOWN ↓"]
    end

    subgraph LENIENT["High Recall (Lenient Model)"]
        direction TB
        B1["Predicts positive aggressively\nFN stays low ✅"] --> B2["Flags real negatives too\nFP rises ❌"] --> B3["Precision goes DOWN ↓"]
    end

    STRICT <-->|"Cannot maximize both\nat the same time"| LENIENT

    style STRICT fill:#2980B9,color:#fff
    style LENIENT fill:#E74C3C,color:#fff
```

---

## 8.4 F1 Score

```
F1 = (2 × Precision × Recall) / (Precision + Recall)
```

```mermaid
graph TD
    P["Precision\n(FP focus)"] --> F1["F1 Score\nHarmonic Mean\nof P and R"]
    R["Recall\n(FN focus)"] --> F1
    F1 --> BAL["Balanced score\nPenalizes extreme imbalance\nbetween P and R"]
    BAL --> WHEN["Use when:\n• Imbalanced dataset\n• Both FP and FN are costly\n• Single number to compare models"]

    style F1 fill:#8E44AD,color:#fff
    style BAL fill:#16A085,color:#fff
```

**Why harmonic mean, not arithmetic?**

| P | R | Arithmetic mean | F1 (Harmonic) | Verdict |
|---|---|----------------|---------------|---------|
| 1.0 | 0.01 | 0.505 ← misleading | **0.02** ← honest | Model is nearly useless |
| 0.8 | 0.8 | 0.80 | **0.80** | Genuinely good |

---

## 8.5 Accuracy

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

```mermaid
pie title All Predictions (Balanced dataset example)
    "True Positive ✅" : 40
    "True Negative ✅" : 45
    "False Positive ❌" : 8
    "False Negative ❌" : 7
```

> Accuracy = (40+45)/(40+45+8+7) = 85/100 = **85%**

**The classic pitfall:**

```mermaid
graph TD
    SKEW["Dataset: 990 healthy, 10 sick\n(1% disease rate)"]
    SKEW --> LAZY["Lazy model: always predicts healthy"]
    LAZY --> ACC["Accuracy = 990/1000 = 99% ✅ looks great"]
    LAZY --> F1["F1 Score = 0 ❌ catches zero sick people"]
    ACC --> WARN["⚠️ Accuracy is misleading\non imbalanced datasets"]

    style SKEW fill:#E74C3C,color:#fff
    style WARN fill:#F39C12,color:#fff
    style F1 fill:#E74C3C,color:#fff
```

---

## 8.6 Sensitivity & Specificity

```mermaid
graph LR
    CM["Confusion Matrix"] --> ROW1["Row 1: Actual Positives\nTP + FN"]
    CM --> ROW2["Row 2: Actual Negatives\nTN + FP"]

    ROW1 --> SENS["Sensitivity = TP / (TP+FN)\n= Recall = TPR\n\nHow well SICK are caught"]
    ROW2 --> SPEC["Specificity = TN / (TN+FP)\n= TNR\n\nHow well HEALTHY are cleared"]

    SENS <-->|"Inverse trade-off\nSensitivity ↑ → Specificity ↓"| SPEC

    style SENS fill:#27AE60,color:#fff
    style SPEC fill:#2980B9,color:#fff
```

**Medical framing:**

| Metric | Question | Example |
|--------|---------|---------|
| Sensitivity | "Of 100 sick people, how many did the test correctly flag?" | COVID RT-PCR |
| Specificity | "Of 100 healthy people, how many did the test correctly clear?" | COVID RT-PCR |

---

## 8.7 FPR & FNR

```mermaid
graph TD
    subgraph WANT_LOW["Want these LOW"]
        FPR["FPR = FP / (TN+FP)\n= 1 − Specificity\n\nFalse alarm rate"]
        FNR["FNR = FN / (TP+FN)\n= 1 − Sensitivity\n\nMiss rate"]
    end

    subgraph WANT_HIGH["Want these HIGH"]
        TPR["TPR = TP / (TP+FN)\n= Sensitivity = Recall\n\nHit rate"]
        TNR["TNR = TN / (TN+FP)\n= Specificity\n\nCorrect rejection rate"]
    end

    FPR <-->|"1 - each other"| TNR
    FNR <-->|"1 - each other"| TPR

    style FPR fill:#E74C3C,color:#fff
    style FNR fill:#E74C3C,color:#fff
    style TPR fill:#27AE60,color:#fff
    style TNR fill:#27AE60,color:#fff
```

---

## 9. ROC Curve & AUC

### What the ROC curve is

```mermaid
graph LR
    T["Vary decision threshold\nfrom 0 to 1"] --> POINT["At each threshold:\ncompute TPR and FPR"]
    POINT --> PLOT["Plot all (FPR, TPR) points\n→ ROC Curve"]
    PLOT --> AUC["Compute area under\nthe curve → AUC"]
    AUC --> JUDGE["Judge classifier quality"]

    style T fill:#2980B9,color:#fff
    style AUC fill:#8E44AD,color:#fff
    style JUDGE fill:#27AE60,color:#fff
```

### Classifier Quality on ROC

```mermaid
graph TD
    subgraph ROC["ROC Space  (FPR on x-axis, TPR on y-axis)"]
        IDEAL["★ Ideal Point\nFPR=0, TPR=1\n(top-left corner)"]
        BEST["Strong Classifier\nAUC ≈ 0.9\ncurve hugs top-left"]
        GOOD["Good Classifier\nAUC ≈ 0.8"]
        RANDOM["Random Classifier\nAUC = 0.5\ndiagonal line"]
        WORSE["Worse than Random\nAUC < 0.5"]
    end

    IDEAL --> BEST --> GOOD --> RANDOM --> WORSE

    style IDEAL fill:#27AE60,color:#fff
    style BEST fill:#2ECC71,color:#fff
    style GOOD fill:#F39C12,color:#fff
    style RANDOM fill:#95A5A6,color:#fff
    style WORSE fill:#E74C3C,color:#fff
```

### AUC & Class Overlap

```mermaid
graph TD
    OV0["No overlap\nNeg: ████  Pos:     ████\nAUC ≈ 1.0 — Perfect"] --> OV1
    OV1["Slight overlap\nNeg: ████\nPos:   ████\nAUC ≈ 0.85 — Good"] --> OV2
    OV2["Moderate overlap\nNeg:  ████\nPos: ████\nAUC ≈ 0.7 — Fair"] --> OV3
    OV3["Full overlap\nNeg: ████\nPos: ████\nAUC = 0.5 — Useless\n(random classifier)"]

    style OV0 fill:#27AE60,color:#fff
    style OV1 fill:#2ECC71,color:#fff
    style OV2 fill:#F39C12,color:#fff
    style OV3 fill:#E74C3C,color:#fff
```

### AUC Quality Table

| AUC | Quality |
|-----|---------|
| 1.0 | Perfect (theoretical) |
| 0.9 – 1.0 | Excellent |
| 0.8 – 0.9 | Good |
| 0.7 – 0.8 | Fair |
| 0.5 | Useless — random |

---

## 10. Complete Cheat Sheet

| Metric | Formula | High value means | Use when |
|--------|---------|-----------------|----------|
| **Precision** | TP / (TP+FP) | Few false alarms | FP is costly |
| **Recall** | TP / (TP+FN) | Few real positives missed | FN is costly |
| **F1 Score** | 2PR / (P+R) | Good P–R balance | Imbalanced data |
| **Accuracy** | (TP+TN) / Total | Overall correctness | Balanced data |
| **Sensitivity** | TP / (TP+FN) | Catches sick people well | Medical +ve diagnosis |
| **Specificity** | TN / (TN+FP) | Clears healthy people well | Medical −ve clearance |
| **FPR** | FP / (TN+FP) | *(want LOW)* | ROC x-axis |
| **FNR** | FN / (TP+FN) | *(want LOW)* | Safety systems |
| **AUC-ROC** | Area under ROC | Better class separation | Comparing classifiers |

---

## 11. Metric Selection — Decision Flowchart

```mermaid
flowchart TD
    START["What kind of evaluation\ndo you need?"] --> Q1{"Dataset\nbalanced?"}

    Q1 -->|Yes| ACC["✅ Use ACCURACY\n(TP+TN)/Total"]
    Q1 -->|No / Skewed| Q2{"Which error is\nmore costly?"}

    Q2 -->|False Positives| PREC["✅ Maximize PRECISION\nTP/(TP+FP)\n\nSpam filter\nLegal conviction\nUnnecessary surgery"]
    Q2 -->|False Negatives| REC["✅ Maximize RECALL\nTP/(TP+FN)\n\nDisease detection\nFraud detection\nSafety systems"]
    Q2 -->|Both equally costly| F1["✅ Use F1 SCORE\n2PR/(P+R)\n\nRecommendation systems\nCancer screening"]

    F1 --> TUNE{"Need to tune\ndecision threshold?"}
    REC --> TUNE
    PREC --> TUNE
    TUNE -->|Yes| ROC["✅ Plot ROC Curve\nOptimize AUC\nPick threshold\nfrom TPR vs FPR tradeoff"]
    TUNE -->|No| DONE["Report chosen metric"]

    style START fill:#2980B9,color:#fff
    style ACC fill:#27AE60,color:#fff
    style PREC fill:#8E44AD,color:#fff
    style REC fill:#D35400,color:#fff
    style F1 fill:#16A085,color:#fff
    style ROC fill:#C0392B,color:#fff
```

---

## 12. Real-World Application Matrix

| Domain | Positive Class | FP consequence | FN consequence | Best Metric |
|--------|---------------|----------------|----------------|------------|
| Email spam | Spam | Legit email blocked | Spam in inbox | Precision |
| COVID test | Infected | Healthy quarantined | Infected walks free | Recall + F1 |
| Cancer screening | Malignant | Unnecessary biopsy | Cancer missed | Recall |
| Fraud detection | Fraud | Transaction declined | Fraud approved | Recall |
| Legal conviction | Guilty | Innocent jailed | Guilty goes free | Precision |
| Factory fault | Equipment fault | Unnecessary shutdown | Fault undetected | Recall |
| Recommendation | Relevant item | Irrelevant shown | Relevant missed | F1 Score |

---

## 13. Key Identities

```
Sensitivity  =  Recall  =  TPR  =  1 − FNR
Specificity  =  TNR          =  1 − FPR

Precision ↑  →  Recall ↓            (strict vs lenient — inverse)
Sensitivity ↑  →  Specificity ↓     (same trade-off, clinical vocabulary)

F1 Score  =  harmonic mean of Precision and Recall
AUC-ROC   =  integral of TPR vs FPR across all thresholds
```

---

*Next: Classification algorithms — Decision Trees, Naive Bayes, SVM — and hyperparameter tuning to shift these metrics in the desired direction.*
