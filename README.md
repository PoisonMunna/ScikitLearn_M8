# 📗 Module 8: Model Evaluation — Learning Notes

> **Goal**: Measure model performance correctly, understand the trade-offs between different evaluation metrics, and diagnose model behavior (bias vs. variance) to guide iterative improvements.

---

## 📌 Table of Contents

1. [Regression Metrics](#1-regression-metrics)
2. [Classification Metrics](#2-classification-metrics)
3. [Confusion Matrix](#3-confusion-matrix)
4. [Classification Report](#4-classification-report)
5. [ROC Curve](#5-roc-curve)
6. [Precision-Recall Curve](#6-precision-recall-curve)
7. [Learning Curve](#7-learning-curve)
8. [Validation Curve](#8-validation-curve)
9. [Error Analysis](#9-error-analysis)

---

## 1. Regression Metrics

### What are Regression Metrics?
Metrics used to evaluate models that predict continuous numerical values. They quantify the difference between predicted values and actual ground truth.

### Key Metrics
- **MAE (Mean Absolute Error)**: Average absolute difference. Robust to outliers. Easy to interpret.
- **MSE (Mean Squared Error)**: Average squared difference. Penalizes large errors heavily.
- **RMSE (Root Mean Squared Error)**: Square root of MSE. Same units as the target variable, making it highly interpretable.
- **$R^2$ (R-squared)**: Represents the proportion of variance in the dependent variable explained by the model. (1.0 is perfect, 0.0 is no better than predicting the mean).

### Code Example

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Generate synthetic regression data
X, y = make_regression(n_samples=1000, n_features=10, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = Ridge(alpha=1.0, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE:  {mae:.2f}")
print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2:   {r2:.4f}")
```

---

## 2. Classification Metrics

### What are Classification Metrics?
Metrics used to evaluate models predicting discrete categories. Unlike regression, they rely on counting correct/incorrect predictions (True/False Positives/Negatives).

### Key Metrics
- **Accuracy**: Overall correctness. *Fails on imbalanced datasets.*
- **Precision**: Out of all predicted positives, how many were actually positive? (Minimizes False Positives).
- **Recall (Sensitivity)**: Out of all actual positives, how many did we find? (Minimizes False Negatives).
- **F1-Score**: Harmonic mean of Precision and Recall. Balances both concerns.

### Code Example

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Generate imbalanced synthetic data
X, y = make_classification(n_samples=1000, weights=[0.9, 0.1], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
```

---

## 3. Confusion Matrix

### What is a Confusion Matrix?
A tabular summary of prediction results, showing the counts of True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).

### Why Use It?
It provides a granular view of *how* the model is failing. For example, in medical diagnosis, a False Negative (missing a disease) is often much worse than a False Positive (false alarm).

### Code Example

```python
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Plotting the confusion matrix
fig, ax = plt.subplots(figsize=(6, 4))
ConfusionMatrixDisplay.from_estimator(
    clf, X_test, y_test, 
    cmap='Blues', 
    display_labels=['Negative (0)', 'Positive (1)'],
    ax=ax
)
plt.title("Confusion Matrix")
plt.grid(False)
plt.show()
```

---

## 4. Classification Report

### What is a Classification Report?
A comprehensive text summary that calculates Precision, Recall, F1-Score, and Support (number of occurrences) for *each* class in a classification problem.

### Why Use It?
It is the standard way to evaluate multiclass models and instantly reveals class-level imbalances and per-class performance drops.

### Code Example

```python
from sklearn.metrics import classification_report

# Generate multiclass data for a better report example
X_multi, y_multi = make_classification(n_samples=1000, n_classes=3, n_informative=4, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

clf_multi = RandomForestClassifier(random_state=42).fit(X_tr, y_tr)
y_pred_multi = clf_multi.predict(X_te)

print(classification_report(y_te, y_pred_multi, target_names=['Class A', 'Class B', 'Class C']))
```

---

## 5. ROC Curve

### What is the ROC Curve?
The **Receiver Operating Characteristic** curve plots the True Positive Rate (Recall) against the False Positive Rate at various classification thresholds.

### Key Concepts
- **AUC (Area Under Curve)**: A single scalar value summarizing the curve. 0.5 is random guessing, 1.0 is perfect.
- **When to use**: Evaluating binary classifiers, especially when classes are roughly balanced. It is threshold-agnostic.

### Code Example

```python
from sklearn.metrics import RocCurveDisplay

fig, ax = plt.subplots(figsize=(7, 5))
RocCurveDisplay.from_estimator(
    clf, X_test, y_test, 
    ax=ax, 
    color='darkgreen',
    name=f'RandomForest (AUC = {RocCurveDisplay.from_estimator(clf, X_test, y_test).roc_auc:.2f})'
)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 6. Precision-Recall Curve

### What is the Precision-Recall Curve?
Plots Precision against Recall at various thresholds. 

### ROC vs. PR Curve
- **ROC** can be overly optimistic on highly imbalanced datasets because the True Negative rate dominates the False Positive Rate calculation.
- **PR Curve** focuses entirely on the positive class. It is the **gold standard for highly imbalanced datasets**.

### Code Example

```python
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score

fig, ax = plt.subplots(figsize=(7, 5))
PrecisionRecallDisplay.from_estimator(clf, X_test, y_test, ax=ax, color='darkred')
plt.axhline(y=y_test.mean(), color='gray', linestyle='--', label='Baseline (Prevalence)')
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 7. Learning Curve

### What is a Learning Curve?
Plots the model's training and cross-validated testing scores as a function of the **number of training samples**.

### Diagnosing Bias vs. Variance
- **High Bias (Underfitting)**: Both training and validation errors are high and converge closely. *Fix: Add features, reduce regularization, use a complex model.*
- **High Variance (Overfitting)**: Training error is low, but validation error is high with a large gap between them. *Fix: Get more data, add regularization, reduce features.*

### Code Example

```python
from sklearn.model_selection import learning_curve

# Calculate learning curve data
train_sizes, train_scores, val_scores = learning_curve(
    RandomForestClassifier(n_estimators=100, random_state=42), 
    X_multi, y_multi, 
    cv=5, 
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)

# Plotting
train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, marker='o', label='Training Score', color='blue')
plt.plot(train_sizes, val_mean, marker='s', label='Validation Score', color='orange')
plt.fill_between(train_sizes, np.mean(train_scores, axis=1) - np.std(train_scores, axis=1), 
                 np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), alpha=0.1, color='blue')
plt.fill_between(train_sizes, np.mean(val_scores, axis=1) - np.std(val_scores, axis=1), 
                 np.mean(val_scores, axis=1) + np.std(val_scores, axis=1), alpha=0.1, color='orange')

plt.title("Learning Curve (Diagnosing Bias/Variance)")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy Score")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 8. Validation Curve

### What is a Validation Curve?
Similar to a learning curve, but instead of varying the *dataset size*, it varies a **specific hyperparameter** (e.g., `max_depth`, `C`, `alpha`) to find its optimal value.

### Why Use It?
It helps pinpoint the exact hyperparameter value where the model transitions from underfitting to overfitting.

### Code Example

```python
from sklearn.model_selection import validation_curve

# Varying the max_depth of a Decision Tree
param_range = np.arange(1, 20)
train_scores, val_scores = validation_curve(
    RandomForestClassifier(n_estimators=50, random_state=42), 
    X_multi, y_multi, 
    param_name="max_depth", 
    param_range=param_range,
    cv=5, 
    scoring='accuracy'
)

train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.figure(figsize=(8, 5))
plt.plot(param_range, train_mean, label='Training Score', color='blue')
plt.plot(param_range, val_mean, label='Validation Score', color='orange')
plt.title("Validation Curve (Tuning max_depth)")
plt.xlabel("max_depth")
plt.ylabel("Accuracy Score")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 9. Error Analysis

### What is Error Analysis?
The qualitative and quantitative investigation of *where* and *why* a model makes mistakes. It goes beyond aggregate metrics to inspect individual predictions.

### Key Steps
1. **Identify Worst Predictions**: Sort errors by magnitude (regression) or confidence (classification).
2. **Look for Patterns**: Do errors cluster around specific subgroups, feature ranges, or data sources?
3. **Check for Data Issues**: Mislabelled data, data leakage, or feature drift.

### Code Example: Residual Analysis for Regression

```python
# Re-using the Ridge Regression model from Section 1
residuals = y_test - y_pred

# Create a dataframe for easier analysis
import pandas as pd
analysis_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred,
    'Residual': residuals,
    'Abs_Error': np.abs(residuals)
})

# Find the top 5 worst predictions
worst_predictions = analysis_df.nlargest(5, 'Abs_Error')
print("--- Top 5 Worst Predictions ---")
print(worst_predictions)

# Plot Residuals vs Predicted (Checking for Homoscedasticity)
plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, alpha=0.5, color='purple')
plt.axhline(y=0, color='red', linestyle='--')
plt.title("Residuals vs Predicted Values")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals (Actual - Predicted)")
plt.grid(True, alpha=0.3)
plt.show()
```

---

### Metric & Tool Selection Guide

| Scenario / Goal | Recommended Metric / Tool | Why? |
|-----------------|---------------------------|------|
| Standard continuous prediction | **RMSE, MAE** | RMSE penalizes large errors; MAE is robust to outliers. |
| Explaining variance in continuous data | **$R^2$ Score** | Tells you how much better your model is than just predicting the mean. |
| Balanced binary classification | **Accuracy, ROC-AUC** | Standard metrics; ROC-AUC evaluates threshold independence. |
| Highly imbalanced classification | **F1-Score, PR-AUC** | Accuracy is misleading; PR-AUC focuses strictly on the minority class. |
| Understanding exact error types | **Confusion Matrix** | Visualizes False Positives vs False Negatives directly. |
| Diagnosing Underfitting vs Overfitting | **Learning Curve** | Shows if the model needs more data (variance) or more complexity (bias). |
| Tuning a specific hyperparameter | **Validation Curve** | Isolates the effect of one hyperparameter on train vs. val performance. |
| Finding edge cases and data flaws | **Error Analysis** | Inspecting individual worst-case predictions reveals hidden data patterns. |

---

### ✅ Module 8 Checklist
- [x] Calculate and interpret MAE, MSE, RMSE, and $R^2$ for regression tasks
- [x] Calculate Accuracy, Precision, Recall, and F1-Score for classification tasks
- [x] Explain why Accuracy is a poor metric for imbalanced datasets
- [x] Generate and interpret a Confusion Matrix to identify FP and FN errors
- [x] Generate a Classification Report for multiclass evaluation
- [x] Plot and interpret an ROC Curve and calculate AUC for balanced data
- [x] Plot and interpret a Precision-Recall Curve for imbalanced data
- [x] Use Learning Curves to diagnose high bias (underfitting) vs high variance (overfitting)
- [x] Use Validation Curves to find optimal hyperparameter values
- [x] Perform residual analysis and error inspection to uncover hidden model flaws
