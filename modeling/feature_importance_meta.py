# ======================================================
# feature_importance_meta.py
# ======================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.exceptions import NotFittedError

from modeling.train_stacked_model import build_model
from modeling.data_load import load_data


# ------------------------------------------------------
# Compute feature importance for a single model
# ------------------------------------------------------
def compute_importance(model, X_val, y_val, feature_names, model_name=""):
    """
    Returns a pandas Series of importances indexed by feature_names.
    Uses native feature_importances_ if available; otherwise permutation importance.
    """
    try:
        # Native tree-based importance
        if hasattr(model, "feature_importances_"):
            return pd.Series(model.feature_importances_, index=feature_names)

        # Fallback to permutation importance
        perm = permutation_importance(
            model,
            X_val,
            y_val,
            n_repeats=5,
            n_jobs=-1,
            random_state=42,
        )
        return pd.Series(perm.importances_mean, index=feature_names)

    except NotFittedError:
        print(f"[WARN] Base model '{model_name}' is not fitted. Skipping its importance.")
        return None


# ------------------------------------------------------
# Compute meta-weighted feature importance and plot
# ------------------------------------------------------
def compute_meta_importance(top_k=20):
    """
    Builds and fits the stacked model, then:
      - extracts fitted base estimators from the StackingClassifier
      - computes per-model feature importances (skipping rf if needed)
      - combines them using absolute meta-learner weights
      - saves CSV and PNG
    Returns:
      combined_importance (Series),
      per_model_importance (dict of Series),
      meta_weights (dict of name -> coefficient)
    """

    # ---------------- Load data ----------------
    X, y, target_col, ids = load_data()
    feature_names = X.columns

    # ---------------- Fit stacked pipeline ----------------
    stacked = build_model(y)
    stacked.fit(X, y)

    # Pipeline parts
    scaler = stacked.named_steps.get("standardscaler", None)
    stack = stacked.named_steps["stackingclassifier"]

    # Use scaled data if scaler exists
    if scaler is not None:
        X_used = scaler.transform(X)
    else:
        X_used = X.values

    # ---------------- Extract fitted estimators ----------------
    # names from original estimators (tuples)
    model_names = [name for name, _ in stack.estimators]
    # fitted versions
    fitted_base_models = stack.estimators_

    # Meta learner: logistic regression
    meta = stack.final_estimator_
    meta_coef = meta.coef_.ravel()  # binary classification

    meta_weights = dict(zip(model_names, meta_coef))

    print("\n===== Meta Learner Coefficients =====")
    for n, w in meta_weights.items():
        print(f"{n}: {w:.4f}")

    # ---------------- Per-model importances ----------------
    per_model_importance = {}
    print("\n===== Computing Base Model Importances =====")

    for name, base_model in zip(model_names, fitted_base_models):
        # You said skipping rf is acceptable
        if name == "rf":
            print("Skipping 'rf' for feature importance as requested.")
            continue

        print(f"Computing importance for: {name}")
        imp_series = compute_importance(
            base_model,
            X_used,
            y,
            feature_names,
            model_name=name,
        )

        if imp_series is not None:
            per_model_importance[name] = imp_series

    if not per_model_importance:
        raise RuntimeError("No base model importances were computed. Check configuration.")

    # ---------------- Combine using meta weights ----------------
    combined = None
    for name, series in per_model_importance.items():
        weight = abs(meta_weights.get(name, 0.0))
        if weight == 0.0:
            # If meta weight is zero, it contributes nothing
            continue

        weighted_series = series * weight

        if combined is None:
            combined = weighted_series.copy()
        else:
            combined += weighted_series

    if combined is None:
        raise RuntimeError("Combined importance is empty. All meta weights may be zero.")

    combined = combined.sort_values(ascending=False)

    # ---------------- Save CSV ----------------
    combined.to_csv("feature_importance_meta_weighted.csv")
    print("\n[Saved] feature_importance_meta_weighted.csv")

    # ---------------- Visualization ----------------
    top_features = combined.head(top_k)

    plt.figure(figsize=(12, 7))
    top_features.plot(kind="bar")
    plt.title("Top Meta Weighted Feature Importances", fontsize=16)
    plt.ylabel("Importance Score", fontsize=14)
    plt.xlabel("Feature", fontsize=12)
    plt.tight_layout()
    plt.savefig("meta_feature_importance.png")
    plt.close()

    print("[Saved] meta_feature_importance.png")

    return combined, per_model_importance, meta_weights


if __name__ == "__main__":
    compute_meta_importance()
