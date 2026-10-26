# Customer Churn Prediction POC (scikit-learn + XGBoost)

An end-to-end proof of concept that predicts whether a customer will churn in the
upcoming year. It generates a realistic synthetic dataset, trains an XGBoost
classifier through a scikit-learn pipeline, and produces a detailed model report
with plots and metrics.

## What it does

1. **Generates synthetic data** (`data/churn_data.csv`) with **41 features** across
   demographics, contract/billing, services, usage, engagement and financial
   behaviour. The target is sampled from a latent logistic risk model so the
   features carry genuine (noisy) signal rather than being random noise.
2. **Pre-processes** categoricals with one-hot encoding (`ColumnTransformer`) and
   keeps numeric features as-is (XGBoost is scale invariant).
3. **Runs 5-fold stratified cross-validation** on the training split.
4. **Trains a final `XGBClassifier`** with early stopping and
   `scale_pos_weight` to handle class imbalance.
5. **Evaluates on a held-out test set** and writes:
   - `reports/model_report.md` — full Markdown report
   - `reports/metrics.json` — machine-readable metrics
   - `reports/confusion_matrix.png`
   - `reports/roc_pr_curves.png`
   - `reports/feature_importance.png`
   - `reports/shap_summary.png` (when `shap` is installed)
   - `reports/churn_model.joblib` — fitted serving pipeline (preprocessor + model)
     so you can score new raw customers with `joblib.load(...).predict_proba(df)`

## Metrics reported

F1, precision, recall, accuracy, balanced accuracy, ROC-AUC, PR-AUC (average
precision), log loss, Brier score, Matthews correlation coefficient, Cohen's
kappa, per-class classification report and the confusion matrix — both on the
held-out test set and as cross-validated mean ± std.

> **A note on R²:** R² is a regression metric and is not meaningful for binary
> classification. The report therefore includes the classification equivalents:
> **D² deviance R²** (`sklearn.metrics.d2_log_loss_score`) and
> **McFadden pseudo-R²**.

## Setup

Requires Python 3.10+. On macOS, XGBoost also needs the OpenMP runtime:

```bash
brew install libomp
```

Create a virtual environment and install dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python churn_poc.py
```

Everything is regenerated from scratch on each run (fixed random seed = 42, so
results are reproducible).

## Tuning knobs

All configuration lives near the top of `churn_poc.py`:

| Constant | Meaning |
|---|---|
| `N_ROWS` | number of synthetic customers (default 20,000) |
| `TEST_SIZE` | held-out test fraction (default 0.20) |
| `N_FOLDS` | cross-validation folds (default 5) |
| `TARGET_CHURN_RATE` | desired churn prevalence (default 0.25) |
| `TOP_N_FEATURES` | features shown in the importance chart (default 25) |
| `SEED` | random seed for reproducibility |
