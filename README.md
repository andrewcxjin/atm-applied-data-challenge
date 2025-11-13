# ATM Applied Data Challenge  
## Machine Learning Pipeline for Ranking-Based Prediction and Model Interpretation

This repository implements an end-to-end machine learning workflow for predicting user behavior from financial transaction data. The project includes data loading, model training, evaluation using ranked precision metrics, calibration analysis, feature importance extraction, and full interpretability reporting.

The workflow is implemented through:
- `train_and_evaluate_pipeline.py`
- `modeling/` directory (modeling logic, evaluation utils, and feature importance modules)
- `eda.ipynb` for exploratory data analysis
- `model_prediction.ipynb` (the notebook summarized below)

The goal is to produce a deployable stacked ensemble model with strong ranking performance and interpretable feature contributions.

---

# 1. Project Overview

This project trains a **stacked classification model** using:
- Random Forest  
- XGBoost  
- LightGBM  
- RBF SVM  
- Logistic Regression meta-learner

The model predicts a binary target representing user-level outcomes derived from financial features.  
Instead of evaluating only accuracy or AUROC, the project emphasizes **ranked performance**, which is crucial in real-world scenarios where only a subset of users can be reviewed or acted on.

Key outcomes:
- Precision at risk cutoffs (5, 10, 20 percent, etc.)
- Probability-bin precision (calibration insight)
- Cumulative ranking curves
- Feature importance across all base models
- Meta-weighted global feature importance
- Holdout predictions CSV for downstream use

---

# 2. Training and Evaluation Pipeline

The main notebook begins by executing:

```python
X, y, target_col, ids = load_data()

model = build_model(y)
model, X_test, y_test, y_pred, y_proba = train_and_evaluate(model, X, y)
```

This pipeline performs:
- Stratified train/test splitting  
- 5-fold cross-validation on the training set  
- Final model fitting  
- Holdout performance reporting  
- AUROC and precision metrics  
- Inspection of the logistic regression meta-learner weights  

The holdout predictions (`y_proba`) become the input to the ranked-evaluation block.

---

# 3. Ranked Precision Evaluation

To evaluate the model as a **ranking system**, the notebook computes:

```python
curve = ranked_precision_at_cutoffs(
    y_test,
    y_proba,
    (0.05, 0.1, 0.2, 0.3, 0.5)
)
display(curve)
```

### What this measures  
Precision@k answers:

> “If we review only the top k percent of highest-scoring users, how many of them are actually positive?”

Cutoffs such as 5, 10, 20 percent correspond to real capacity limits in fraud, credit risk, medical triage, and similar domains.

The notebook also computes precision by probability bin:

```python
bins = ranked_precision_by_bins(y_test, y_proba)
display(bins)
```

This groups predictions into ranges such as:
- 0.0–0.1  
- 0.1–0.2  
- …  
- 0.9–1.0  

and reports actual precision inside each bin.  
This reveals whether higher probabilities truly correspond to higher empirical risk.

---

# 4. Precision Plots

Two plots are produced:

### **Cumulative Precision Curve**
```python
plot_cumulative_precision(curve, title="Cumulative Precision (Holdout)")
```

Shows how precision decays as we expand from the top 5 percent to larger parts of the distribution.  
A steep curve indicates strong concentration of signal among top-ranked users.

### **Probability-Bin Precision Plot**
```python
plot_bin_precision(bins, title="Precision by Probability Bin (Holdout)")
```

Shows model calibration:  
Higher bins should contain higher true positive rates.

---

# 5. Exporting Predictions for Downstream Use

The notebook exports a holdout-set predictions file with aligned user IDs:

```python
save_predictions_csv(
    model=model,
    X=X_test,
    ids_df=ids_test,
    y_true=y_test,
    path="model_predictions_holdout.csv",
)
```

This enables additional analysis, auditing, or downstream integration.

---

# 6. Feature Importance Analysis

### Per-Model Feature Importances
We compute feature importances for each base estimator using:

```python
from modeling.feature_importance import main as run_feature_importance
importance_results = run_feature_importance()
```

This provides:
- Native tree-based importances for RF, XGB, LGBM  
- Permutation importance for SVC  
- Sorted tables for each individual model  

### Meta-Weighted Global Importance

A second module computes a combined importance score that reflects how the meta-learner uses each model’s predictions.

```python
from modeling.feature_importance_meta import compute_meta_importance

combined_importance, per_model_importance, meta_weights = compute_meta_importance()
```

Outputs include:
- A global ranking of features weighted by meta-learner coefficients  
- Individual importance plots for each model  
- A final summary plot of the top 20 features  
- CSV and PNG exports  

This provides a holistic view of which input variables truly drive the final ensemble’s decisions.

---

# 7. Exploratory Data Analysis (EDA)

`eda.ipynb` includes:
- Initial data integrity checks  
- Distributions of major transaction features  
- Outlier detection  
- Correlation heatmaps  
- Target balance and stratification checks  
- Missingness and feature sparsity analysis  
- Early feature importance heuristics  

The insights from EDA informed:
- Feature scaling decisions  
- Handling of class imbalance  
- Model selection and tuning  

---

# 8. Repository Structure

```
project/
│
├── eda.ipynb
├── model_prediction.ipynb
│
├── modeling/
│   ├── data_load.py
│   ├── train_stacked_model.py
│   ├── evaluation_utils.py
│   ├── feature_importance.py
│   ├── feature_importance_meta.py
│
├── train_and_evaluate_pipeline.py
├── model_predictions_holdout.csv
├── README.md
```

---

# 9. Key Takeaways

- The model performs strongly as a **ranking system**, with high precision in the top risk deciles.
- Cumulative and binned precision curves confirm a well-ordered probability distribution and useful calibration.
- Ensemble modeling with a logistic regression meta-learner provides both performance and interpretability.
- Feature importance analyses (per-model and meta-weighted) identify which financial behaviors most strongly predict the target outcome.
- The full workflow is reproducible, extensible, and ready for operational deployment or research analysis.

---

# 10. How to Run

### Environment
```
pip install -r requirements.txt
```

### Execute the notebook
Open `model_prediction.ipynb` and run all cells.

### Or run the pipeline script
```
python train_and_evaluate_pipeline.py
```

---

# 11. License
MIT License (or your preferred license).
