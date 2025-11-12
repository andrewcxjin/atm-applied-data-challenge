# ======================================================
# train_stacked_model.py
# ======================================================
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score, roc_auc_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import pyarrow
import fastparquet


# ======================================================
# Build Model
# ======================================================
def build_model(y_train):
    """Build a stacked ensemble model with tree-based and kernel-based classifiers,
    and a logistic regression meta-learner for calibrated blending.
    """

    # Handle imbalance ratio
    pos_ratio = (y_train == 1).sum() / max(len(y_train), 1)
    scale_pos_weight = (1 - pos_ratio) / max(pos_ratio, 1e-6)

    rf = ("rf", RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=42,
    ))

    xgb = ("xgb", XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        tree_method="hist",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=42
    ))

    lgbm = ("lgbm", LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    ))

    svc = ("svc", make_pipeline(
        StandardScaler(with_mean=False),
        SVC(
            kernel="rbf",
            C=2.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=42,
        ),
    ))

    meta = LogisticRegression(
        solver="lbfgs",
        max_iter=3000,
        class_weight="balanced",
        random_state=42,
    )

    stack = StackingClassifier(
        estimators=[rf, xgb, lgbm, svc],
        final_estimator=meta,
        stack_method="predict_proba",
        n_jobs=-1,
    )

    model = make_pipeline(StandardScaler(with_mean=False), stack)
    return model


# ======================================================
# Train & Evaluate
# ======================================================
def train_and_evaluate(model, X, y, test_size=0.2, random_state=42):
    """Train the stacked model, perform 5-fold cross-validation,
    and evaluate precision and AUROC on a held-out set.

    Returns:
        model, X_test, y_test, y_pred, y_pred_proba
    """

    # Suppress harmless model warnings for cleaner output
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    warnings.filterwarnings("ignore", message="Parameters: {\"use_label_encoder\"}")
    warnings.filterwarnings("ignore", category=UserWarning)

    # ---- Split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # ---- Cross-validation ----
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring={"precision": "precision", "auroc": "roc_auc"},
        n_jobs=-1,
        return_train_score=False,
    )

    print("\nCross-Validation Metrics (mean ± std):")
    print(f"  Precision: {cv_results['test_precision'].mean():.3f} ± {cv_results['test_precision'].std():.3f}")
    print(f"  AUROC:     {cv_results['test_auroc'].mean():.3f} ± {cv_results['test_auroc'].std():.3f}")

    # ---- Fit final model ----
    model.fit(X_train, y_train)

    # ---- Evaluate on holdout ----
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    auroc = roc_auc_score(y_test, y_pred_proba)

    print("\nHoldout Evaluation:")
    print(f"  Precision: {precision:.3f}")
    print(f"  AUROC:     {auroc:.3f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, digits=3))

    # ---- Meta-learner diagnostics ----
    try:
        stack_obj = model.named_steps["stackingclassifier"]
        final_lr = stack_obj.final_estimator_
        names = [n for n, _ in stack_obj.estimators]
        print("\nMeta-learner Coefficients:")
        for n, c in zip(names, final_lr.coef_.ravel()):
            print(f"  {n:<6}: {c:+.3f}")
    except Exception:
        pass

    return model, X_test, y_test, y_pred, y_pred_proba