# ======================================================
# feature_importance.py
# ======================================================
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

from modeling.train_stacked_model import build_model
from modeling.data_load import load_data

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# ======================================================
# Utility: save importance table
# ======================================================
def save_importance(df, name):
    df.to_csv(f"feature_importance_{name}.csv", index=False)
    print(f"[Saved] feature_importance_{name}.csv")


# ======================================================
# Extract native feature importance if exists
# ======================================================
def get_native_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        return pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
    return None


# ======================================================
# Compute permutation importance (for SVC, LR meta, etc.)
# ======================================================
def get_permutation_importance(model, X_val, y_val, feature_names, n_repeats=7):
    perm = permutation_importance(
        model, X_val, y_val,
        n_repeats=n_repeats,
        n_jobs=-1,
        random_state=42
    )
    return pd.DataFrame({
        "feature": feature_names,
        "importance": perm.importances_mean
    }).sort_values("importance", ascending=False)


# ======================================================
# Train each base model individually
# ======================================================
def evaluate_single_model(model, model_name, X_train, X_val, y_train, y_val, feature_names):
    print(f"\n=== Evaluating {model_name} ===")

    # Fit
    model.fit(X_train, y_train)

    # Native importance
    native = get_native_importance(model, feature_names)

    if native is not None:
        print(f"Native importances found for {model_name}.")
        save_importance(native, f"{model_name}_native")
    else:
        # Compute permutation for models without native importance
        print(f"No native importances for {model_name}. Using permutation importance.")
        perm = get_permutation_importance(model, X_val, y_val, feature_names)
        save_importance(perm, f"{model_name}_perm")
        native = perm

    return native


# ======================================================
# Main
# ======================================================
def main():
    # ---- Load data ----
    X, y, target_col, ids = load_data()
    feature_names = X.columns

    # ---- Split ----
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # ---- Base models (mirroring your stack) ----
    rf = RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=42,
    )

    xgb = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42
    )

    lgbm = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

    svc = make_pipeline(
        StandardScaler(with_mean=False),
        SVC(
            kernel="rbf",
            C=2.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=42
        )
    )

    # ---- Evaluate and save importances ----
    importance_results = {
        "rf": evaluate_single_model(rf, "rf", X_train, X_val, y_train, y_val, feature_names),
        "xgb": evaluate_single_model(xgb, "xgb", X_train, X_val, y_train, y_val, feature_names),
        "lgbm": evaluate_single_model(lgbm, "lgbm", X_train, X_val, y_train, y_val, feature_names),
        "svc": evaluate_single_model(svc, "svc", X_train, X_val, y_train, y_val, feature_names),
    }

    print("\nDone. Feature importance files saved.")
    return importance_results


if __name__ == "__main__":
    main()
