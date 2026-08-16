# 📚 Library & Function Reference

Notebook: `colab_notebook.ipynb` — Auto MPG EDA & Imputation Lab

---

## 1. `pandas` (`pd`)

Core library for tabular data manipulation.

| Function / Attribute | Usage in Notebook | Description |
|---|---|---|
| `pd.read_csv(path)` | Load dataset | Reads a CSV file into a DataFrame |
| `df.shape` | Check dimensions | Returns `(rows, cols)` tuple |
| `df.info()` | Inspect schema | Prints dtypes, non-null counts, memory usage |
| `df.columns` | Iterate columns | Returns column name index |
| `df[col].unique()` | Explore cardinality | Returns array of distinct values in a column |
| `df.copy()` | Safe duplication | Creates an independent copy — prevents modifying the original |
| `df.loc[rows, cols]` | Inject NaNs / assign values | Label-based row+column indexing and assignment |
| `df.head(n)` | Spot-check | Returns first `n` rows |
| `df[col].astype(str)` | Type casting | Converts column to string; needed to use as categorical `hue` in seaborn |
| `df[cols].corr()` | Correlation matrix | Pairwise Pearson correlation for numeric columns |
| `df[col].map({...})` | Recode values | Maps numeric codes to labels (e.g. `1 → "USA"`) |
| `df.fillna(value)` | Manual imputation | Fills NaN entries with a provided scalar or Series |
| `df[col].mode().iloc[0]` | Most-frequent value | Returns the mode; `.iloc[0]` picks the first if there's a tie |
| `pd.DataFrame(data, columns, index)` | Wrap NumPy array | Converts imputer output array back to a DataFrame with original index |

---

## 2. `numpy` (`np`)

Numerical computing — random sampling and constants.

| Function / Attribute | Usage in Notebook | Description |
|---|---|---|
| `np.random.seed(42)` | Reproducibility | Seeds the random number generator so results are deterministic |
| `np.random.choice(index, size, replace)` | Inject missing values | Randomly samples row indices to set as NaN |
| `np.nan` | Sentinel value | IEEE 754 Not-a-Number; used to represent missing data |

---

## 3. `warnings`

Standard library module to control runtime warnings.

| Function | Usage in Notebook | Description |
|---|---|---|
| `warnings.filterwarnings('ignore')` | Suppress noise | Hides deprecation and convergence warnings during exploration |

---

## 4. `sklearn.impute.SimpleImputer`

Fill missing values with a statistical strategy.

| Function / Attribute | Usage in Notebook | Description |
|---|---|---|
| `SimpleImputer(strategy='mean')` | Mean imputation | Replaces NaN with column mean |
| `SimpleImputer(strategy='median')` | Median imputation | Replaces NaN with column median — robust to outliers |
| `SimpleImputer(strategy='most_frequent')` | Mode imputation | Replaces NaN with most frequent value in the column |
| `imputer.fit_transform(X)` | Fit + apply in one step | Learns statistics from `X`, then transforms it |

> **When to use what:** Mean — when distribution is roughly symmetric. Median — when there are outliers. Mode — for categorical or skewed numeric data. Class-based mode — when the target label is available and distributions differ by class.

---

## 5. `sklearn.decomposition.PCA`

Principal Component Analysis — dimensionality reduction and variance inspection.

| Function / Attribute | Usage in Notebook | Description |
|---|---|---|
| `PCA()` | Initialize | No `n_components` → keeps all components (used to inspect full variance) |
| `pca.fit_transform(X_scaled)` | Fit & project | Learns principal components then projects data into PC space |
| `pca.explained_variance_ratio_` | Scree analysis | Array of variance fractions explained by each PC |
| `pca.components_` | Loadings matrix | Shape `(n_components, n_features)`; each row is a PC direction |
| `pca.n_components_` | Component count | Number of PCs retained after fit |

> **Key insight used in notebook:** `pca.components_.T` transposes loadings to shape `(n_features, n_components)` for a readable feature-contribution table.

---

## 6. `sklearn.preprocessing.StandardScaler`

Standardize features to zero mean, unit variance — required before PCA.

| Function | Usage in Notebook | Description |
|---|---|---|
| `StandardScaler()` | Initialize | No parameters needed for default z-score scaling |
| `scaler.fit_transform(X)` | Scale data | Computes mean & std from `X`, then scales in one step |

---

## 7. `matplotlib.pyplot` (`plt`)

Low-level plotting — figure/axes creation and layout control.

| Function | Usage in Notebook | Description |
|---|---|---|
| `plt.figure(figsize=(w, h))` | Create figure | Opens a new blank figure with specified size |
| `plt.subplots(rows, cols, figsize)` | Grid of axes | Returns `(fig, axes)` — use `axes.flatten()` for easy iteration |
| `axes.flatten()` | Flatten 2D axes array | Converts `[[ax, ax], [ax, ax]]` → `[ax, ax, ax, ax]` for indexed access |
| `plt.suptitle(text, y, fontsize)` | Figure-level title | Title above all subplots; `y > 1.0` raises it above tight layout |
| `plt.tight_layout()` | Prevent overlap | Auto-adjusts spacing between subplots |
| `plt.show()` | Render | Displays the current figure |
| `plt.scatter(x, y, alpha, c, label)` | Biplot scatter | Plots PCA-projected points coloured by class |
| `plt.legend(title, labels, loc, bbox_to_anchor)` | External legend | `bbox_to_anchor=(1, 0.5)` places legend to the right of the plot |
| `plt.grid()` | Grid lines | Adds reference grid to the current axes |
| `axes[i].set_title()` | Per-subplot title | Sets title on a specific axes object |
| `axes[i].set_xlabel()` / `.set_ylabel()` | Axis labels | Labels for x and y axes on a specific subplot |

---

## 8. `seaborn` (`sns`)

High-level statistical visualisation built on matplotlib.

| Function | Usage in Notebook | Description |
|---|---|---|
| `sns.set(style="whitegrid")` | Global style | Sets background style; `"whitegrid"` adds subtle grid lines |
| `sns.countplot(data, x, color)` | Categorical bar chart | Counts occurrences of each category; good for `cylinders`, `origin`, `model_year` |
| `sns.pairplot(df, vars, diag_kind, hue, palette)` | Feature matrix | Grid of scatter plots for all feature pairs; diagonal shows KDE or histogram |
| `pairplot._legend.remove()` | Clean up legend | Removes seaborn's auto-placed legend so a manual one can be positioned outside |
| `sns.heatmap(data, annot, cmap, center, fmt)` | Correlation heatmap | Visualises correlation matrix; `annot=True` prints values; `cmap='coolwarm'` diverges at 0 |
| `sns.histplot(data, x, bins, kde, hue, multiple, ax)` | Histogram | `multiple='stack'` stacks class bars; `kde=True` overlays density curve |
| `sns.kdeplot(data, x, fill, hue, common_norm, ax)` | Density plot | `common_norm=False` normalises each class independently — reveals shape differences |
| `sns.boxplot(data, x, y, ax)` | Box plot | Shows median, IQR, whiskers, and outliers; use `x=class_col` for side-by-side |
| `sns.violinplot(data, x, y, inner, palette, ax)` | Violin plot | Combines KDE shape with boxplot statistics; `inner='box'` shows box inside the violin |

---

## Quick-Reference: Import Block

```python
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
```

---

## Workflow Summary

```
Load CSV (pandas)
    ↓
Explore shape / info / unique counts (pandas)
    ↓
Inject synthetic NaNs (numpy.random)
    ↓
Impute missing values — mean / median / mode / class-mode (SimpleImputer + pandas)
    ↓
Scale features (StandardScaler)
    ↓
Run PCA — inspect variance & loadings (PCA)
    ↓
Visualise distributions (seaborn: countplot, pairplot, heatmap, histplot, kdeplot, boxplot, violinplot)
```
