# Classification with Decision Trees — Session Notes
**Dataset**: Telco Customer Churn | **Algorithm**: Decision Tree Classifier & Random Forest  
**Libraries**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

---

## 1. Problem Statement

**Telco Customer Churn** is a binary classification problem:

- **Dataset**: Tabular data from a fictitious telecom company (~7,043 rows, 21 columns)
- **Target variable**: `Churn` — whether a customer left the service (`Yes`/`No`)
- **Input features**: Demographics, service usage, account details
- **Goal**: Predict whether a customer will churn based on their features

### Feature Types

| Type | Examples |
|------|----------|
| **Categorical** | Gender, SeniorCitizen, Partner, Dependents, PhoneService, InternetService, Contract, etc. |
| **Numeric** | Tenure (months), MonthlyCharges, TotalCharges |

> **Note**: `SeniorCitizen` holds values `0` and `1` — looks numeric but is categorical (binary flag, not a measurement).

---

## 2. Python Libraries Used

| Library | Purpose |
|---------|---------|
| `pandas` | Loading CSV files, data manipulation (DataFrames) |
| `matplotlib` | Plotting graphs (bar charts, line plots) |
| `seaborn` | Heatmaps and statistical visualizations |
| `numpy` | Mathematical operations, array handling |
| `scikit-learn (sklearn)` | Everything ML: preprocessing, train-test split, models, metrics |

### Key sklearn Modules

```python
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, classification_report)
```

---

## 3. Exploratory Data Analysis (EDA)

### 3.1 Loading the Dataset

```python
import pandas as pd

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(df)     # Shows head and tail rows
df.info()     # Column names, inferred data types, non-null counts
```

`df.info()` tells you:
- Total row and column count
- The dtype pandas assigned to each column (important — pandas may guess wrong)
- Non-null count per column (to spot missing values)

---

### 3.2 Observation 1 — Hidden Missing Values in `TotalCharges`

**Problem**: `TotalCharges` should be a numeric column (float) but pandas infers it as `object` (string).

**Why?** Some rows have **empty strings `" "`** instead of a real number or `NaN`. Since they are not `NaN`, pandas does not flag them as missing — it treats the whole column as text.

```python
# Find the rows with blank TotalCharges
mask = df["TotalCharges"].str.strip() == ""
print(df[mask][["customerID", "tenure", "TotalCharges"]])
# 11 rows with blank TotalCharges
```

**Fix**: Force conversion to numeric. Blanks become `NaN`, then drop those rows.

```python
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)
# Row count: 7043 -> 7032 (11 rows removed)
```

**Key concept — `errors='coerce'`**: When converting to numeric, any value that cannot be parsed (like an empty string) is silently replaced with `NaN` instead of raising an error.

---

### 3.3 Observation 2 — `SeniorCitizen` Should Be Categorical

**Problem**: `SeniorCitizen` is stored as `int64` because its only values are `0` and `1`. But it represents a *category* (is senior citizen or not), not a quantity.

**Fix**: Explicitly change the dtype to `object` (pandas dtype for categorical/string data):

```python
df["SeniorCitizen"] = df["SeniorCitizen"].astype("object")
```

---

### 3.4 Observation 3 — Redundant "No" Categories

Columns like `MultipleLines`, `OnlineSecurity`, `OnlineBackup`, `TechSupport`, `StreamingTV`, `StreamingMovies` have **three** unique values instead of two:

| Column | Values Present |
|--------|---------------|
| `MultipleLines` | `Yes`, `No`, `No phone service` |
| `OnlineSecurity` | `Yes`, `No`, `No internet service` |
| `TechSupport` | `Yes`, `No`, `No internet service` |

`No phone service` and `No internet service` are both just forms of "No" for our prediction purpose.

**Fix**: Replace the verbose "No" variants with plain `"No"`:

```python
cols_no_internet = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies"]
cols_no_phone = ["MultipleLines"]

for col in cols_no_internet:
    df[col] = df[col].replace("No internet service", "No")
for col in cols_no_phone:
    df[col] = df[col].replace("No phone service", "No")
```

All these columns now have only two values: `Yes` / `No`.

---

### 3.5 Separating Feature Types

```python
cat_cols = df.select_dtypes(include="object").columns         # 18 categorical columns
num_cols = df.select_dtypes(include=["int64","float64"]).columns  # 3 numeric columns
```

---

### 3.6 Correlation / Association Heatmaps

#### Numeric Features — Pearson Correlation

```python
df["Churn_num"] = df["Churn"].map({"Yes": 1, "No": 0})
corr_matrix = df[list(num_cols) + ["Churn_num"]].corr()

import seaborn as sns
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
```

**Key findings**:
- `tenure` and `TotalCharges`: strong positive correlation (long-tenure customers pay more cumulatively)
- `tenure` and `Churn`: moderate negative correlation (r ~ -0.35) — loyal customers churn less

#### Categorical Features — Cramer's V

Pearson correlation only works for numeric features — it measures linear relationships between continuous values. It is mathematically undefined for categories. For categorical variables we use **Cramer's V** — a measure of association derived from the chi-squared statistic, ranging from 0 (no association) to 1 (perfect association).

```python
import warnings, numpy as np
from scipy.stats import chi2_contingency

warnings.filterwarnings("ignore")

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k - 1, r - 1))

assoc = pd.DataFrame(index=cat_cols, columns=cat_cols, dtype=float)
for c1 in cat_cols:
    for c2 in cat_cols:
        assoc.loc[c1, c2] = 1.0 if c1 == c2 else cramers_v(df[c1], df[c2])

sns.heatmap(assoc, annot=True, cmap="coolwarm")
```

**Key findings**:
- `customerID` shows inflated association — every value is unique, always drop before modeling
- `Contract` type has ~0.4 association with `Churn`
- `InternetService` correlates with related service columns (~0.44)
- Most categorical features have low direct association with `Churn`

---

## 4. Preprocessing for Modeling

### 4.1 Discretizing Numeric Features (Binning)

Decision trees work cleanest with categorical data. We convert the 3 numeric columns into discrete bins using **quantile-based (equal-frequency) binning**.

**What is quantile binning?**
Sort all values ascending, then divide into k equally-sized groups. Each bin holds the same proportion of data points (~25% each for 4 bins).

```
Sorted values:  [---Bin 0---][---Bin 1---][---Bin 2---][---Bin 3---]
                  0th-25th%   25th-50th%   50th-75th%   75th-100th%
```

```python
from sklearn.preprocessing import KBinsDiscretizer

disc_tenure  = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="quantile")
disc_monthly = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="quantile")
disc_total   = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="quantile")

df["tenure_bin"]          = disc_tenure.fit_transform(df[["tenure"]])
df["MonthlyCharges_bin"]  = disc_monthly.fit_transform(df[["MonthlyCharges"]])
df["TotalCharges_bin"]    = disc_total.fit_transform(df[["TotalCharges"]])
```

> **Why a separate object per column?** Each `KBinsDiscretizer` learns the quantile boundary values of its specific column during `fit`. Reusing the same object for another column would apply wrong boundaries.

---

### 4.2 Dropping Redundant Columns

```python
df2 = df.copy()
df2.drop(columns=["customerID"], inplace=True)
df2.drop(columns=["tenure", "MonthlyCharges", "TotalCharges"], inplace=True)

# Explicitly mark bin columns as categorical
for col in ["tenure_bin", "MonthlyCharges_bin", "TotalCharges_bin"]:
    df2[col] = df2[col].astype("object")
```

After this step: **20 columns** — 19 input features + 1 target (`Churn`), all categorical.

---

### 4.3 Class Distribution — Imbalanced Data

```python
df2["Churn"].value_counts(normalize=True) * 100
# No:  ~73%
# Yes: ~26%
```

**Why this matters**: A naive model that always predicts `No` achieves **73% accuracy** without learning anything. A trained model at 78% is only **5% better than doing nothing**. For imbalanced datasets, use F1, Precision, Recall, and AUC-ROC to measure real learning.

---

## 5. Creating X (Features) and Y (Target)

```python
# Target: map Yes/No to 1/0; strip() removes invisible whitespace
df2["Churn"] = df2["Churn"].str.strip().map({"Yes": 1, "No": 0})
Y = df2["Churn"]

# Features: everything except Churn (19 columns)
X = df2.drop(columns=["Churn"])
```

---

### One-Hot Encoding (OHE)

**Why?** ML models only understand numbers. Assigning integers to categories (`DSL=0, Fiber=1, No=2`) implies a false ordering. One-hot encoding creates a **separate binary column** per category value.

```
InternetService    ->    InternetService_DSL    InternetService_Fiber optic
DSL                            1                          0
Fiber optic                    0                          1
No                             0                          0   <- both 0 implies "No"
```

**`drop_first=True`**: With n categories, n-1 binary columns fully encode the information (the nth is implied when all others are 0). This avoids a redundant column per feature.

```python
X_encoded = pd.get_dummies(X, drop_first=True)
# ~19 original columns expand to ~29 binary columns
```

---

## 6. Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X_encoded, Y,
    test_size=0.1,     # Initial experiment: 90% train, 10% test
    random_state=42,   # Reproducibility — same split every run
    stratify=Y         # Preserve ~73%/26% class ratio in both sets
)
```

**`stratify=Y`**: Without this, random sampling could give the test set very few `Churn=Yes` samples, making evaluation unreliable.

**`random_state=42`**: Seeds the random number generator. Everyone using the same seed gets the same train/test rows — essential for reproducible experiments.

---

## 7. Decision Tree — Hyperparameter Tuning

A **hyperparameter** is a configuration you set before training. The model learns weights from data but cannot learn its own hyperparameters.

For Decision Trees, the primary hyperparameter is **`max_depth`**.

---

### 7.1 Finding Optimal Depth

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

train_accs, test_accs = [], []

for depth in range(1, 30):
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=depth, random_state=42)
    clf.fit(X_train, Y_train)
    train_accs.append(accuracy_score(Y_train, clf.predict(X_train)))
    test_accs.append(accuracy_score(Y_test,  clf.predict(X_test)))

plt.plot(range(1, 30), train_accs, label="Train Accuracy")
plt.plot(range(1, 30), test_accs,  label="Test Accuracy")
plt.xlabel("Max Depth"); plt.ylabel("Accuracy"); plt.legend()
plt.show()
```

**Reading the graph**:
- Train accuracy continuously rises — deeper tree memorizes more
- Test accuracy rises until ~depth 6, then drops — overfitting begins
- **Conclusion**: Use `max_depth=6`

**Visual intuition**:
```
Accuracy
  High |  Train: ───────────────────────────>
       |              Test: ──────────
  Low  |  Test:  /              \_______>
       └─────────────────────────────────────
            1   2   3   4   5   6   7   8 ...
                              ^
                         Best depth (6)
```

---

### 7.2 Finding Optimal Train-Test Split Ratio

With `max_depth=6` fixed, test different split sizes:

```python
split_ratios = [0.6, 0.7, 0.8, 0.9]

for ratio in split_ratios:
    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X_encoded, Y, train_size=ratio, random_state=42, stratify=Y)
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=6, random_state=42)
    clf.fit(X_tr, Y_tr)

    te_pred = clf.predict(X_te)
    te_prob = clf.predict_proba(X_te)[:, 1]  # Probabilities for AUC

    print(f"Train={ratio:.0%}  Acc={accuracy_score(Y_te,te_pred):.3f}  "
          f"F1={f1_score(Y_te,te_pred):.3f}  AUC={roc_auc_score(Y_te,te_prob):.3f}")
```

> **`predict_proba()`** returns probability scores (0.0-1.0) per class rather than hard 0/1 labels. AUC-ROC requires these probabilities to compute its ranking-quality metric.

**Decision**: **80/20 split** performs best or near-best across Accuracy, Recall, and F1. It also gives a reasonable evaluation set size (20% of 7032 ~ 1400 samples).

---

## 8. Training the Final Model

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X_encoded, Y, train_size=0.8, random_state=42, stratify=Y)

clf = DecisionTreeClassifier(criterion="entropy", max_depth=6, random_state=42)
clf.fit(X_train, Y_train)    # <- Model trained here

Y_test_pred = clf.predict(X_test)
Y_test_prob = clf.predict_proba(X_test)[:, 1]

print("Train Accuracy:", accuracy_score(Y_train, clf.predict(X_train)))
print("Test  Accuracy:", accuracy_score(Y_test, Y_test_pred))
print(classification_report(Y_test, Y_test_pred))
```

**Typical results** (depth=6, 80-20 split):
- Train accuracy: ~79%
- Test accuracy:  ~78%
- Small gap confirms no severe overfitting

---

## 9. Classification Metrics — Definitions

Confusion matrix for binary classification:

```
                    Predicted Positive    Predicted Negative
Actual Positive     TP (True Positive)    FN (False Negative)
Actual Negative     FP (False Positive)   TN (True Negative)
```

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Accuracy** | (TP+TN) / Total | Overall fraction correctly classified |
| **Precision** | TP / (TP+FP) | Of all predicted churners, how many actually churned |
| **Recall** | TP / (TP+FN) | Of all actual churners, how many did we correctly catch |
| **F1 Score** | 2*P*R / (P+R) | Harmonic mean of Precision and Recall — balanced metric |
| **AUC-ROC** | Area under ROC curve | 0.5=random, 1.0=perfect; measures ranking quality |

> For imbalanced data, **F1, Recall, and AUC are more informative than raw accuracy**.

---

## 10. Visualizing the Decision Tree

```python
from sklearn.tree import plot_tree

plt.figure(figsize=(28, 14))
plot_tree(
    clf,
    feature_names=X_encoded.columns.tolist(),
    class_names=["No Churn", "Churn"],
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title("Decision Tree — depth=6, criterion=entropy")
plt.show()
```

Each node shows:
- **Feature** used for splitting
- **Entropy** at that node (0 = perfectly pure / all one class)
- **Sample count** passing through
- **Class distribution** at that node (color intensity = purity)

---

## 11. Entropy vs Gini — Split Criteria

At every node the algorithm greedily picks the feature whose split maximally reduces impurity.

| Criterion | Formula | Notes |
|-----------|---------|-------|
| **Entropy** (Information Gain) | -sum(p * log2(p)) | Range [0,1]; slightly slower; penalizes impurity more |
| **Gini Impurity** | 1 - sum(p^2) | Faster to compute; usually gives similar results |

Both converge to the same goal — finding the most discriminative feature at each level.

**Practical effect in this session**:
- `criterion="entropy"` -> root node feature: `Contract` type
- `criterion="gini"`    -> root node feature: `InternetService_Fiber optic`

Try both and compare test metrics; differences are usually small.

---

## 12. Overfitting vs Underfitting

| Scenario | Train Accuracy | Test Accuracy | Diagnosis |
|----------|---------------|--------------|-----------|
| Depth 1-2 | Low (~72%) | Low (~72%) | **Underfitting** — too simple to capture patterns |
| Depth 5-6 | Moderate (~79%) | Moderate (~78%) | **Good fit** |
| Depth 20+ | Very High (~95%) | Low (~73%) | **Overfitting** — memorized training noise |

**The goal**: minimize train-test accuracy gap while maximizing test accuracy.

---

## 13. Random Forest — Overview

Random Forest trains many decision trees and combines predictions via **majority vote**.

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,    # Number of trees (key hyperparameter)
    max_depth=6,
    criterion="entropy",
    random_state=42
)
rf.fit(X_train, Y_train)
print(classification_report(Y_test, rf.predict(X_test)))
```

| | Decision Tree | Random Forest |
|---|---|---|
| Number of trees | 1 | Many (n_estimators) |
| Feature selection | All features considered | Random subset per tree |
| Overfitting risk | High | Lower (averaging reduces variance) |
| Interpretability | High (visualizable) | Low (black-box ensemble) |

**How the "random" part works**: Each tree sees a random subset of features. Different trees learn different aspects of the data. Final prediction = majority vote across all trees.

**Tuning `n_estimators`**: Same approach as depth — plot test accuracy vs [50, 75, 100, 150] trees and pick the point of diminishing returns.

---

## 14. Saving and Deploying the Model

Once trained, the model can be serialized to disk for reuse without retraining:

```python
import pickle

# Save
with open("churn_model.pkl", "wb") as f:
    pickle.dump(clf, f)

# Load and predict on new data
with open("churn_model.pkl", "rb") as f:
    loaded_clf = pickle.load(f)

# New data must go through the same preprocessing pipeline first
prediction = loaded_clf.predict(new_X_encoded)
```

**Production flow**:
```
New customer data (JSON)
         |
         v
Preprocessing pipeline:
  - strip whitespace
  - map Yes/No -> 1/0
  - apply saved KBinsDiscretizer (disc_tenure, disc_monthly, disc_total)
  - apply get_dummies one-hot encoding
         |
         v
loaded_clf.predict(X_processed)
         |
         v
Churn = 0 (stays) or 1 (will leave)
```

> **Data drift**: Real-world data distribution shifts over time. Periodically retrain on newly accumulated data to keep predictions accurate.

---

## 15. Key Takeaways

1. **Always run `df.info()`** — pandas can silently mistype columns (e.g., numeric stored as string due to blank entries)
2. **Missing values are not always NaN** — blank strings look fine to pandas but break numeric conversion; use `errors='coerce'`
3. **Imbalanced data makes accuracy misleading** — a 73% No / 26% Yes split means a dumb model scores 73%; use F1, Recall, AUC
4. **Discretize numeric features** before Decision Trees for cleaner, more interpretable splits
5. **One-hot encode categoricals** with `drop_first=True` — prevents false ordinal ordering; n-1 columns are sufficient for n categories
6. **`stratify=Y`** in train-test split preserves the class imbalance ratio in both splits
7. **`random_state`** seeds the random number generator — ensures reproducibility across runs
8. **Hyperparameters need empirical tuning** — plot metrics across a range of values and pick the peak/elbow
9. **The test set is sacred** — never train on it; it simulates unseen real-world data
10. **Random Forest** reduces overfitting by averaging many trees trained on random feature subsets
