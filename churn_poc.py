"""
Customer Churn Prediction POC (scikit-learn + XGBoost)
======================================================
End-to-end proof of concept:

  1. Generates a realistic synthetic customer-churn dataset (~40 features).
  2. Pre-processes it (one-hot encoding for categoricals, numeric passthrough).
  3. Trains an XGBoost classifier with 5-fold stratified cross-validation and
     early stopping on a validation split.
  4. Evaluates on a held-out test set and writes a detailed model report
     (Markdown + JSON) plus plots:
        - confusion matrix
        - ROC / precision-recall curves
        - top-N feature importance
        - SHAP summary (when the ``shap`` package is available)

Run:
    python churn_poc.py

Outputs:
    data/churn_data.csv               generated synthetic dataset
    reports/model_report.md           human readable model report
    reports/metrics.json              machine readable metrics
    reports/churn_model.joblib        fitted serving pipeline (preproc + model)
    reports/*.png                     plots
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless backend (no display needed)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier

try:  # scikit-learn >= 1.5 provides the deviance R**2 score directly.
    from sklearn.metrics import d2_log_loss_score
except ImportError:  # pragma: no cover - fallback for older scikit-learn
    d2_log_loss_score = None

try:
    import shap

    SHAP_AVAILABLE = True
except Exception:  # pragma: no cover - shap is optional
    SHAP_AVAILABLE = False


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
SEED = 42
N_ROWS = 20_000
TEST_SIZE = 0.20
VALID_SIZE = 0.15
N_FOLDS = 5
TARGET_CHURN_RATE = 0.25
TOP_N_FEATURES = 25

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "reports"

TARGET = "churn"
ID_COLUMN = "customer_id"

# --------------------------------------------------------------------------
# Synthetic data generation
# --------------------------------------------------------------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + 1e-9)


def generate_churn_data(n_rows: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    """Create a realistic but fully synthetic customer-churn dataset.

    The target is sampled from a latent logistic risk model so that the
    features carry genuine (noisy) signal instead of being pure noise.
    """
    rng = np.random.default_rng(seed)

    # ---- Demographics ----------------------------------------------------
    age = rng.integers(18, 76, n_rows)
    gender = rng.choice(["Male", "Female", "Other"], n_rows, p=[0.49, 0.49, 0.02])
    senior_citizen = (age >= 60).astype(int)
    partner = rng.choice(["Yes", "No"], n_rows, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], n_rows, p=[0.30, 0.70])

    # ---- Contract / billing ---------------------------------------------
    contract_type = rng.choice(
        ["Month-to-month", "One year", "Two year"], n_rows, p=[0.55, 0.25, 0.20]
    )
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        n_rows,
        p=[0.34, 0.23, 0.22, 0.21],
    )
    paperless_billing = rng.choice(["Yes", "No"], n_rows, p=[0.59, 0.41])
    tenure_months = rng.integers(0, 73, n_rows)
    base_charge = rng.normal(70, 15, n_rows).clip(18, 130)
    monthly_charges = np.round(
        base_charge + 2.0 * (contract_type == "Month-to-month"), 2
    )
    total_charges = np.round(
        monthly_charges * (tenure_months + 1) * rng.uniform(0.95, 1.05, n_rows), 2
    )

    # ---- Services --------------------------------------------------------
    phone_service = rng.choice(["Yes", "No"], n_rows, p=[0.90, 0.10])
    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No internet service"],
        n_rows,
        p=[0.34, 0.44, 0.22],
    )
    has_internet = internet_service != "No internet service"

    def _service(p_yes: float) -> np.ndarray:
        return np.where(rng.random(n_rows) < p_yes, "Yes", "No")

    multiple_lines = np.where(phone_service == "Yes", _service(0.45), "No")
    online_security = np.where(has_internet, _service(0.35), "No")
    online_backup = np.where(has_internet, _service(0.40), "No")
    device_protection = np.where(has_internet, _service(0.40), "No")
    tech_support = np.where(has_internet, _service(0.35), "No")
    streaming_tv = np.where(has_internet, _service(0.45), "No")
    streaming_movies = np.where(has_internet, _service(0.45), "No")

    # ---- Usage -----------------------------------------------------------
    avg_monthly_gb_download = np.where(
        has_internet, rng.gamma(2.0, 60.0, n_rows).clip(0, 600), 0.0
    )
    avg_monthly_gb_upload = np.where(
        has_internet, rng.gamma(2.0, 12.0, n_rows).clip(0, 150), 0.0
    )
    total_data_usage_gb = np.round(
        (avg_monthly_gb_download + avg_monthly_gb_upload) * np.maximum(tenure_months, 1), 1
    )
    avg_call_minutes = np.where(
        phone_service == "Yes", rng.gamma(3.0, 90.0, n_rows).clip(0, 1500), 0.0
    )
    num_customer_service_calls = rng.poisson(1.2, n_rows).clip(0, 15)
    num_complaints = rng.poisson(0.5, n_rows).clip(0, 10)
    avg_ticket_resolution_days = np.round(rng.gamma(2.0, 3.0, n_rows).clip(0, 30), 2)

    # ---- Engagement ------------------------------------------------------
    num_logins_per_month = rng.poisson(18, n_rows).clip(0, 120)
    app_sessions_per_month = rng.poisson(45, n_rows).clip(0, 400)
    num_transactions_per_month = rng.poisson(8, n_rows).clip(0, 150)
    loyalty_points = (
        tenure_months * rng.uniform(20, 120, n_rows) + rng.normal(0, 300, n_rows)
    ).clip(0, None).round(0)
    email_open_rate = rng.beta(2, 3, n_rows).round(4)
    days_since_last_interaction = rng.integers(0, 366, n_rows)
    discount_used = rng.choice([0, 1], n_rows, p=[0.65, 0.35])
    promotional_offers_accepted = rng.poisson(1.0, n_rows).clip(0, 10)

    # ---- Financial -------------------------------------------------------
    credit_score = rng.normal(680, 90, n_rows).clip(300, 850).round(0)
    annual_income = rng.lognormal(11.0, 0.45, n_rows).clip(15_000, 400_000).round(0)
    late_payment_count = rng.poisson(0.8, n_rows).clip(0, 12)
    refund_requests = rng.poisson(0.3, n_rows).clip(0, 5)
    avg_monthly_spend = np.round(monthly_charges * rng.uniform(0.8, 1.4, n_rows), 2)
    customer_lifetime_value = np.round(
        monthly_charges * np.maximum(tenure_months, 1) * rng.uniform(0.6, 1.2, n_rows) / 100,
        2,
    )

    # ---- Latent churn propensity ----------------------------------------
    logit = (
        0.55 * _zscore(num_complaints)
        + 0.45 * _zscore(num_customer_service_calls)
        + 0.35 * _zscore(avg_ticket_resolution_days)
        + 0.40 * _zscore(monthly_charges)
        + 0.35 * _zscore(late_payment_count)
        + 0.30 * _zscore(days_since_last_interaction)
        + 0.30 * _zscore(refund_requests)
        - 0.80 * _zscore(tenure_months)
        - 0.45 * _zscore(loyalty_points)
        - 0.35 * _zscore(credit_score)
        - 0.30 * _zscore(num_logins_per_month)
        - 0.25 * _zscore(email_open_rate)
        - 0.25 * _zscore(promotional_offers_accepted)
        + 0.60 * (contract_type == "Month-to-month")
        - 0.40 * (contract_type == "Two year")
        + 0.45 * (internet_service == "Fiber optic")
        - 0.30 * (tech_support == "Yes")
        - 0.25 * (online_security == "Yes")
        - 0.20 * (discount_used == 1)
        + 0.25 * senior_citizen
        + 0.20 * (payment_method == "Electronic check")
    )
    logit = logit + rng.normal(0, 0.35, n_rows)  # noise -> not perfectly separable

    # Calibrate the intercept so the churn rate matches TARGET_CHURN_RATE.
    lo, hi = -8.0, 8.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if _sigmoid(logit + mid).mean() < TARGET_CHURN_RATE:
            lo = mid
        else:
            hi = mid
    logit = logit + (lo + hi) / 2.0
    churn = rng.binomial(1, _sigmoid(logit))

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST{i:06d}" for i in range(n_rows)],
            "age": age,
            "gender": gender,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "tenure_months": tenure_months,
            "contract_type": contract_type,
            "payment_method": payment_method,
            "paperless_billing": paperless_billing,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "phone_service": phone_service,
            "multiple_lines": multiple_lines,
            "internet_service": internet_service,
            "online_security": online_security,
            "online_backup": online_backup,
            "device_protection": device_protection,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
            "avg_monthly_gb_download": avg_monthly_gb_download.round(2),
            "avg_monthly_gb_upload": avg_monthly_gb_upload.round(2),
            "total_data_usage_gb": total_data_usage_gb,
            "avg_call_minutes": avg_call_minutes.round(2),
            "num_customer_service_calls": num_customer_service_calls,
            "num_complaints": num_complaints,
            "avg_ticket_resolution_days": avg_ticket_resolution_days,
            "num_logins_per_month": num_logins_per_month,
            "app_sessions_per_month": app_sessions_per_month,
            "num_transactions_per_month": num_transactions_per_month,
            "loyalty_points": loyalty_points,
            "email_open_rate": email_open_rate,
            "days_since_last_interaction": days_since_last_interaction,
            "discount_used": discount_used,
            "promotional_offers_accepted": promotional_offers_accepted,
            "credit_score": credit_score,
            "annual_income": annual_income,
            "late_payment_count": late_payment_count,
            "refund_requests": refund_requests,
            "avg_monthly_spend": avg_monthly_spend,
            "customer_lifetime_value": customer_lifetime_value,
            TARGET: churn,
        }
    )
    return df


# --------------------------------------------------------------------------
# Model building helpers
# --------------------------------------------------------------------------
def split_feature_types(df: pd.DataFrame):
    """Return (feature names, categorical columns, numeric columns)."""
    features = [c for c in df.columns if c not in (TARGET, ID_COLUMN)]
    cat_cols = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in features if c not in cat_cols]
    return features, cat_cols, num_cols


def build_preprocessor(cat_cols, num_cols) -> ColumnTransformer:
    """One-hot encode categoricals, pass numeric columns through untouched.

    XGBoost is tree based and therefore scale invariant, so no scaler is
    needed.  ``handle_unknown="ignore"`` keeps the transformer safe for
    unseen categories at inference time.
    """
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model(scale_pos_weight: float, early_stopping: bool = False) -> XGBClassifier:
    """Create an XGBoost classifier tuned for imbalanced binary churn."""
    params = dict(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=2,
        gamma=0.0,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=SEED,
    )
    if early_stopping:
        params["early_stopping_rounds"] = 50
    return XGBClassifier(**params)


CV_SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "neg_log_loss": "neg_log_loss",
    "matthews": "matthews_corrcoef",
}


def cross_validate_model(X_train, y_train, cat_cols, num_cols, scale_pos_weight):
    """Run stratified k-fold CV on the *raw* training frame (no leakage)."""
    cv_pipeline = Pipeline(
        [
            ("prep", build_preprocessor(cat_cols, num_cols)),
            ("clf", build_model(scale_pos_weight)),
        ]
    )
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return cross_validate(
        cv_pipeline, X_train, y_train, cv=cv, scoring=CV_SCORING, n_jobs=-1
    )


def train_final_model(X_train, y_train, cat_cols, num_cols):
    """Fit the preprocessor + final XGBoost model with early stopping."""
    scale_pos_weight = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1.0)
    preprocessor = build_preprocessor(cat_cols, num_cols)
    X_train_pre = preprocessor.fit_transform(X_train)

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train_pre,
        y_train,
        test_size=VALID_SIZE,
        stratify=y_train,
        random_state=SEED,
    )
    model = build_model(scale_pos_weight, early_stopping=True)
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)

    return {
        "model": model,
        "preprocessor": preprocessor,
        "scale_pos_weight": scale_pos_weight,
        "best_iteration": int(getattr(model, "best_iteration", 0) or 0),
        "best_validation_auc": float(getattr(model, "best_score", float("nan"))),
    }


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Compute the full battery of classification metrics."""
    eps = 1e-12
    p = np.clip(np.asarray(y_proba, dtype=float), eps, 1 - eps)
    y_true = np.asarray(y_true)

    # McFadden pseudo-R**2 (an R**2 analogue for classification).
    ll_model = float(np.sum(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))
    p_null = float(np.clip(np.mean(y_true), eps, 1 - eps))
    ll_null = float(np.sum(y_true * np.log(p_null) + (1 - y_true) * np.log(1 - p_null)))
    mcfadden = 1.0 - ll_model / ll_null if ll_null != 0 else float("nan")

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "mcfadden_pseudo_r2": float(mcfadden),
    }
    if d2_log_loss_score is not None:
        metrics["d2_log_loss"] = float(d2_log_loss_score(y_true, y_proba))
    return metrics


def plot_confusion_matrix(y_true, y_pred, path: Path) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No churn", "Churn"],
        yticklabels=["No churn", "Churn"],
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title("Confusion matrix - test set")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return cm


def plot_curves(y_true, y_proba, path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    baseline = float(np.mean(y_true))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr, color="tab:blue", label=f"ROC-AUC = {roc_auc:.4f}")
    axes[0].plot([0, 1], [0, 1], "--", color="grey", label="Chance")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve")
    axes[0].legend(loc="lower right")

    axes[1].plot(rec, prec, color="tab:green", label=f"PR-AUC = {ap:.4f}")
    axes[1].axhline(baseline, ls="--", color="grey", label=f"Baseline ({baseline:.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-recall curve")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_importance(model, feature_names, path: Path, top_n: int = TOP_N_FEATURES):
    importances = np.asarray(model.feature_importances_, dtype=float)
    order = np.argsort(importances)[::-1][:top_n]
    names = np.asarray(feature_names)

    fig, ax = plt.subplots(figsize=(8.5, max(5.0, 0.32 * len(order) + 1.5)))
    y_pos = np.arange(len(order))
    ax.barh(y_pos, importances[order], color="tab:blue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names[order], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importance (normalised gain)")
    ax.set_title(f"Top {len(order)} feature importances")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [(str(names[i]), float(importances[i])) for i in order]


def plot_shap_summary(model, X_transformed, feature_names, path: Path):
    if not SHAP_AVAILABLE:
        return None
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = _shap_as_positive_class(explainer.shap_values(X_transformed))
        shap.summary_plot(
            shap_values,
            X_transformed,
            feature_names=list(feature_names),
            show=False,
            max_display=20,
        )
        fig = plt.gcf()
        fig.set_size_inches(9, 8)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"  ! SHAP plot skipped: {exc}")
        return None


def _shap_as_positive_class(shap_values):
    """Normalise the many shapes returned by different SHAP versions."""
    if isinstance(shap_values, list):  # some versions return [class0, class1]
        return shap_values[1] if len(shap_values) > 1 else shap_values[0]
    values = np.asarray(shap_values)
    if values.ndim == 3:  # (n_samples, n_features, n_classes)
        return values[:, :, 1]
    return values


# --------------------------------------------------------------------------
# Report generation
# --------------------------------------------------------------------------
TEST_METRIC_LABELS = [
    ("f1", "F1 score (churn class)"),
    ("precision", "Precision (churn class)"),
    ("recall", "Recall / sensitivity (churn class)"),
    ("accuracy", "Accuracy"),
    ("balanced_accuracy", "Balanced accuracy"),
    ("roc_auc", "ROC-AUC"),
    ("pr_auc", "PR-AUC (average precision)"),
    ("log_loss", "Log loss"),
    ("brier_score", "Brier score"),
    ("matthews_corrcoef", "Matthews correlation coefficient"),
    ("cohen_kappa", "Cohen's kappa"),
    ("d2_log_loss", "D2 deviance R^2 (classification R^2 analogue)"),
    ("mcfadden_pseudo_r2", "McFadden pseudo-R^2"),
]

CV_METRIC_LABELS = [
    ("f1", "F1 score"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("accuracy", "Accuracy"),
    ("balanced_accuracy", "Balanced accuracy"),
    ("roc_auc", "ROC-AUC"),
    ("pr_auc", "PR-AUC"),
    ("neg_log_loss", "Log loss (negated)"),
    ("matthews", "Matthews correlation coefficient"),
]


def _fmt(value) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def write_report(
    path: Path,
    dataset_summary: dict,
    model_info: dict,
    test_metrics: dict,
    cv_results: dict,
    cm: np.ndarray,
    class_report: str,
    feature_importance: list,
    plot_paths: dict,
) -> None:
    """Render the model report as Markdown."""
    cv_mean = {k[5:]: float(np.mean(v)) for k, v in cv_results.items() if k.startswith("test_")}
    cv_std = {k[5:]: float(np.std(v)) for k, v in cv_results.items() if k.startswith("test_")}

    lines = [
        "# Customer Churn Prediction - Model Report",
        "",
        "> Generated by `churn_poc.py` using **scikit-learn** and **XGBoost**.",
        "",
        "Binary target: `churn` (1 = customer churns in the upcoming year, 0 = stays).",
        "",
        "## 1. Dataset",
        "",
        f"- Rows: **{dataset_summary['n_rows']:,}**",
        f"- Features: **{dataset_summary['n_features']}** "
        f"({dataset_summary['n_numeric']} numeric, {dataset_summary['n_categorical']} categorical)",
        f"- Overall churn rate: **{dataset_summary['churn_rate']:.2%}**",
        f"- Train / test split: **{1 - TEST_SIZE:.0%} / {TEST_SIZE:.0%}** (stratified)",
        f"- Cross-validation: **{N_FOLDS}-fold** stratified on the training split",
        "",
        "## 2. Model",
        "",
        "| Parameter | Value |",
        "|---|---:|",
        f"| Algorithm | XGBClassifier |",
        f"| n_estimators | {model_info['n_estimators']} |",
        f"| learning_rate | {model_info['learning_rate']} |",
        f"| max_depth | {model_info['max_depth']} |",
        f"| min_child_weight | {model_info['min_child_weight']} |",
        f"| subsample | {model_info['subsample']} |",
        f"| colsample_bytree | {model_info['colsample_bytree']} |",
        f"| reg_lambda | {model_info['reg_lambda']} |",
        f"| scale_pos_weight | {model_info['scale_pos_weight']:.4f} |",
        f"| early_stopping_rounds | {model_info['early_stopping_rounds']} |",
        f"| best_iteration | {model_info['best_iteration']} |",
        f"| best validation ROC-AUC | {model_info['best_validation_auc']:.4f} |",
        "",
        "## 3. Held-out test set performance",
        "",
        "| Metric | Score |",
        "|---|---:|",
    ]
    for key, label in TEST_METRIC_LABELS:
        if key in test_metrics:
            lines.append(f"| {label} | {_fmt(test_metrics[key])} |")

    lines += [
        "",
        "## 4. Cross-validated performance (training split)",
        "",
        "| Metric | Mean | Std. dev. |",
        "|---|---:|---:|",
    ]
    for key, label in CV_METRIC_LABELS:
        if key in cv_mean:
            lines.append(f"| {label} | {cv_mean[key]:.4f} | {cv_std[key]:.4f} |")

    lines += [
        "",
        "## 5. Confusion matrix (test set)",
        "",
        "|  | Predicted: no churn | Predicted: churn |",
        "|---|---:|---:|",
        f"| **Actual: no churn** | {int(cm[0, 0]):,} (TN) | {int(cm[0, 1]):,} (FP) |",
        f"| **Actual: churn** | {int(cm[1, 0]):,} (FN) | {int(cm[1, 1]):,} (TP) |",
        "",
        "## 6. Per-class classification report",
        "",
        "```text",
        class_report.rstrip(),
        "```",
        "",
        "## 7. Top feature importances",
        "",
        "| Rank | Feature | Importance |",
        "|---:|---|---:|",
    ]
    for rank, (name, importance) in enumerate(feature_importance, start=1):
        lines.append(f"| {rank} | `{name}` | {importance:.4f} |")

    lines += ["", "## 8. Plots", ""]
    for title, image in plot_paths.items():
        if image:
            lines.append(f"### {title}")
            lines.append("")
            lines.append(f"![{title}]({Path(image).name})")
            lines.append("")

    lines += [
        "## 9. A note on R^2",
        "",
        "R^2 (the coefficient of determination) is defined for **continuous regression** "
        "targets. Churn is a **binary classification** problem, so a plain R^2 is not a "
        "valid measure here. The closest classification equivalents are reported above:",
        "",
        "- **D2 deviance R^2** - `1 - log_loss(model) / log_loss(null model)`. "
        "1.0 is a perfect model, 0.0 is no better than always predicting the base rate, "
        "negative values are worse than the base rate.",
        "- **McFadden pseudo-R^2** - a likelihood-ratio analogue of R^2.",
        "",
        "## 10. Key takeaways",
        "",
        f"- The model reaches an F1 of **{test_metrics['f1']:.4f}**, precision of "
        f"**{test_metrics['precision']:.4f}** and recall of **{test_metrics['recall']:.4f}** "
        "on the held-out test set.",
        f"- ROC-AUC of **{test_metrics['roc_auc']:.4f}** indicates strong ranking ability.",
        "- `scale_pos_weight` compensates for class imbalance; for production, tune the "
        "decision threshold against the business cost of false positives vs. false negatives.",
        "- SHAP values (if available) explain *why* each customer is scored as high risk, "
        "which is essential for retention teams.",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Customer Churn Prediction POC (scikit-learn + XGBoost)")
    print("=" * 78)

    # 1. Data --------------------------------------------------------------
    print("\n[1/6] Generating synthetic churn dataset ...")
    df = generate_churn_data()
    data_path = DATA_DIR / "churn_data.csv"
    df.to_csv(data_path, index=False)
    features, cat_cols, num_cols = split_feature_types(df)
    churn_rate = float(df[TARGET].mean())
    print(
        f"      rows={len(df):,} | features={len(features)} "
        f"(numeric={len(num_cols)}, categorical={len(cat_cols)}) | churn={churn_rate:.2%}"
    )
    print(f"      saved -> {data_path}")

    # 2. Split -------------------------------------------------------------
    print("\n[2/6] Splitting data (stratified train/test) ...")
    X = df[features]
    y = df[TARGET].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    print(f"      train={len(X_train):,} | test={len(X_test):,}")

    # 3. Cross-validation --------------------------------------------------
    scale_pos_weight = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1.0)
    print(f"\n[3/6] Running {N_FOLDS}-fold stratified cross-validation ...")
    cv_results = cross_validate_model(
        X_train, y_train, cat_cols, num_cols, scale_pos_weight
    )
    print(
        f"      CV ROC-AUC = {np.mean(cv_results['test_roc_auc']):.4f} "
        f"+/- {np.std(cv_results['test_roc_auc']):.4f}"
    )
    print(
        f"      CV F1      = {np.mean(cv_results['test_f1']):.4f} "
        f"+/- {np.std(cv_results['test_f1']):.4f}"
    )

    # 4. Final model -------------------------------------------------------
    print("\n[4/6] Training final XGBoost model with early stopping ...")
    trained = train_final_model(X_train, y_train, cat_cols, num_cols)
    model = trained["model"]
    preprocessor = trained["preprocessor"]
    feature_names = list(preprocessor.get_feature_names_out())
    print(
        f"      scale_pos_weight={trained['scale_pos_weight']:.4f} | "
        f"best_iteration={trained['best_iteration']} | "
        f"best_val_auc={trained['best_validation_auc']:.4f}"
    )

    # 5. Evaluate ----------------------------------------------------------
    print("\n[5/6] Evaluating on the held-out test set ...")
    X_test_pre = preprocessor.transform(X_test)
    y_proba = model.predict_proba(X_test_pre)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    test_metrics = compute_metrics(y_test, y_pred, y_proba)
    class_report = classification_report(
        y_test, y_pred, target_names=["No churn", "Churn"], digits=4
    )
    for key, label in TEST_METRIC_LABELS:
        if key in test_metrics:
            print(f"      {label:<48s} {test_metrics[key]:.4f}")

    # 6. Plots + report ----------------------------------------------------
    print("\n[6/6] Writing plots and model report ...")
    plot_paths = {}

    cm_path = REPORT_DIR / "confusion_matrix.png"
    cm = plot_confusion_matrix(y_test, y_pred, cm_path)
    plot_paths["Confusion matrix"] = str(cm_path)

    curves_path = REPORT_DIR / "roc_pr_curves.png"
    plot_curves(y_test, y_proba, curves_path)
    plot_paths["ROC and precision-recall curves"] = str(curves_path)

    importance_path = REPORT_DIR / "feature_importance.png"
    feature_importance = plot_feature_importance(model, feature_names, importance_path)
    plot_paths[f"Top {TOP_N_FEATURES} feature importances"] = str(importance_path)

    shap_path = REPORT_DIR / "shap_summary.png"
    shap_result = plot_shap_summary(model, X_test_pre, feature_names, shap_path)
    plot_paths["SHAP summary (test set)"] = str(shap_result) if shap_result else None

    params = model.get_params()
    model_info = {
        "n_estimators": params["n_estimators"],
        "learning_rate": params["learning_rate"],
        "max_depth": params["max_depth"],
        "min_child_weight": params["min_child_weight"],
        "subsample": params["subsample"],
        "colsample_bytree": params["colsample_bytree"],
        "reg_lambda": params["reg_lambda"],
        "scale_pos_weight": trained["scale_pos_weight"],
        "early_stopping_rounds": params.get("early_stopping_rounds", 0),
        "best_iteration": trained["best_iteration"],
        "best_validation_auc": trained["best_validation_auc"],
    }
    dataset_summary = {
        "n_rows": int(len(df)),
        "n_features": len(features),
        "n_numeric": len(num_cols),
        "n_categorical": len(cat_cols),
        "churn_rate": churn_rate,
    }

    report_path = REPORT_DIR / "model_report.md"
    write_report(
        path=report_path,
        dataset_summary=dataset_summary,
        model_info=model_info,
        test_metrics=test_metrics,
        cv_results=cv_results,
        cm=cm,
        class_report=class_report,
        feature_importance=feature_importance,
        plot_paths=plot_paths,
    )

    # Machine-readable metrics alongside the report.
    cv_summary = {
        key[5:]: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "folds": [float(v) for v in values],
        }
        for key, values in cv_results.items()
        if key.startswith("test_")
    }
    metrics_path = REPORT_DIR / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "dataset": dataset_summary,
                "test_metrics": test_metrics,
                "cross_validation": cv_summary,
                "model": model_info,
                "feature_importance_top": [
                    {"feature": name, "importance": importance}
                    for name, importance in feature_importance
                ],
                "confusion_matrix": cm.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Persist a ready-to-use serving pipeline (fitted preprocessor + model).
    serving_model = Pipeline([("prep", preprocessor), ("clf", model)])
    model_path = REPORT_DIR / "churn_model.joblib"
    joblib.dump(serving_model, model_path)

    print("\n      Sample predictions on held-out customers:")
    sample = X_test.head(5)
    customer_ids = df.loc[sample.index, ID_COLUMN]
    sample_proba = serving_model.predict_proba(sample)[:, 1]
    for customer_id, probability in zip(customer_ids, sample_proba):
        verdict = "CHURN" if probability >= 0.5 else "stay"
        print(f"        {customer_id}: churn_probability={probability:.3f} -> {verdict}")

    print(f"\n      report   -> {report_path}")
    print(f"      metrics  -> {metrics_path}")
    print(f"      model    -> {model_path}")
    print(f"      plots    -> {REPORT_DIR}")
    print("\nDone.")


if __name__ == "__main__":
    main()

