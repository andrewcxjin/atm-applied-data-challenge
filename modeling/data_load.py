# ------------------ Robust loaders + normalizers ------------------
import os
import ast
import numpy as np
import pandas as pd
from datetime import timedelta

# ---------- Helpers ----------

def _read_parquet(path):
    """Force pyarrow engine to ensure consistency."""
    return pd.read_parquet(path, engine="pyarrow")

def _to_dt(s, utc=False):
    """Convert to datetime safely (keeps tz if already datetime)."""
    if not pd.api.types.is_datetime64_any_dtype(s):
        s = pd.to_datetime(s, errors="coerce", utc=utc)
    return s

def _decode_text(val):
    """Decode bytes to UTF-8 text (fallback latin-1), else str()."""
    if isinstance(val, (bytes, bytearray)):
        try:
            return val.decode("utf-8")
        except Exception:
            try:
                return val.decode("latin-1")
            except Exception:
                return val.hex()
    return val

def _decode_possible_bytestr(s):
    """Convert stringified byte literals like "b'\\x12\\x34'" → hex."""
    if isinstance(s, str) and s.startswith("b'") and s.endswith("'"):
        try:
            b = ast.literal_eval(s)
            if isinstance(b, (bytes, bytearray)):
                return b.hex()
        except Exception:
            pass
    elif isinstance(s, (bytes, bytearray)):
        return s.hex()
    if pd.isna(s):
        return np.nan
    return str(s)

def _normalize_id_cols(df, id_cols):
    """Normalize ID columns into comparable hex strings."""
    for c in id_cols:
        if c in df.columns:
            df[c] = df[c].apply(_decode_possible_bytestr)
    return df

def _normalize_text_cols(df, text_cols):
    """Ensure text columns are properly decoded to str."""
    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].apply(_decode_text)
    return df

def _ensure_numeric(df):
    """Keep only numeric columns, coerce invalids, fill inf/nan."""
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            try:
                df[c] = pd.to_numeric(df[c])
            except Exception:
                continue
    num = df.select_dtypes(include=[np.number]).copy()
    num = num.replace([np.inf, -np.inf], np.nan).fillna(0)
    return num


# ------------------ Main Data Loader ------------------

def load_data():
    print("=== Loading datasets ===")

    # ---- 1) Advances / First Advances ----
    if os.path.exists("data/first_advances.csv"):
        first_advances = pd.read_csv("data/first_advances.csv")
        first_advances = _normalize_id_cols(first_advances, ["request_id"])
    else:
        advances_df = _read_parquet("data/advances.parquet")
        advances_df = _normalize_id_cols(advances_df, ["request_id", "user_id"])
        advances_df["underwritten_at"] = _to_dt(advances_df["underwritten_at"])

        try:
            advances_df["user_id"] = pd.to_numeric(advances_df["user_id"])
        except Exception:
            pass

        advances_sorted = advances_df.sort_values(["user_id", "underwritten_at"])
        first_advances = advances_sorted.groupby("user_id", as_index=False).first()
        first_advances.to_csv("data/first_advances.csv", index=False)

    print(f"first_advances: {len(first_advances)} rows")

    # ---- 2) Labels ----
    labels_df = _read_parquet("data/labels.parquet")
    labels_df = _normalize_id_cols(labels_df, ["advance_id"])
    print(f"labels: {len(labels_df)} rows")

    # ---- 3) Transactions ----
    tx_paths = [
        "data/transactions_sample.parquet",
        "data/transactions_sample_2.parquet",
        "data/transactions_sample_3.parquet",
        "data/transactions_sample_4.parquet",
    ]
    tx_frames = []
    for p in tx_paths:
        if os.path.exists(p):
            t = _read_parquet(p)
            t = _normalize_id_cols(t, ["user_id", "id"])
            t = _normalize_text_cols(t, ["merchant_name", "primary_category", "detailed_category"])
            tx_frames.append(t)

    transactions_all = pd.concat(tx_frames, ignore_index=True)
    transactions_all["date"] = _to_dt(transactions_all["date"])
    transactions_all["amount"] = pd.to_numeric(transactions_all["amount"], errors="coerce").fillna(0)

    try:
        transactions_all["user_id"] = pd.to_numeric(transactions_all["user_id"])
    except Exception:
        pass

    print(f"transactions_all: {len(transactions_all)} rows, {transactions_all['user_id'].nunique()} users")

    # ---- 4) Spending Percentages ----
    if os.path.exists("data/spending_pct.csv"):
        spending_pct = pd.read_csv("data/spending_pct.csv")
    else:
        tx_sorted = transactions_all.sort_values(["user_id", "date"])
        categories = tx_sorted["primary_category"].dropna().astype(str).unique().tolist()
        for rm in ["TRANSFER_IN", "INCOME"]:
            if rm in categories:
                categories.remove(rm)

        spending = tx_sorted[
            (tx_sorted["primary_category"].isin(categories)) & (tx_sorted["amount"] > 0)
        ].copy()

        if spending.empty:
            spending_pct = pd.DataFrame({"user_id": pd.unique(first_advances["user_id"])})
            spending_pct["total_spending"] = 0.0
        else:
            spending_agg = (
                spending.groupby(["user_id", "primary_category"])["amount"].sum().reset_index()
            )
            spending_wide = spending_agg.pivot(
                index="user_id", columns="primary_category", values="amount"
            ).fillna(0)
            spending_wide["total_spending"] = spending_wide.sum(axis=1)
            spending_pct = spending_wide.copy()
            denom = spending_pct["total_spending"].replace(0, 1)
            for col in categories:
                spending_pct[col] = spending_pct.get(col, 0) / denom
            spending_pct = spending_pct.reset_index().fillna(0)
        spending_pct.to_csv("data/spending_pct.csv", index=False)

    print(f"spending_pct: {len(spending_pct)} rows")

    # ---- 5) Flow Metrics ----
    if os.path.exists("data/flow_metrics.csv"):
        flow_metrics = pd.read_csv("data/flow_metrics.csv")
    else:
        tx_sorted = transactions_all.sort_values(["user_id", "date"])
        inflow_cats = ["INCOME", "TRANSFER_IN"]
        inflow_tx = tx_sorted[
            (tx_sorted["primary_category"].isin(inflow_cats)) & (tx_sorted["amount"] < 0)
        ][["user_id", "amount"]]
        inflow_agg = inflow_tx.groupby("user_id")["amount"].sum().abs().reset_index(name="total_inflow")

        income_agg = (
            tx_sorted.query("primary_category == 'INCOME' and amount < 0")
            .groupby("user_id")["amount"].sum().abs().reset_index(name="total_income")
        )
        transfer_agg = (
            tx_sorted.query("primary_category == 'TRANSFER_IN' and amount < 0")
            .groupby("user_id")["amount"].sum().abs().reset_index(name="total_transfer_in")
        )
        outflow_agg = spending_pct[["user_id", "total_spending"]].rename(
            columns={"total_spending": "total_outflow"}
        )

        flow_metrics = (
            outflow_agg.merge(inflow_agg, on="user_id", how="outer")
            .merge(income_agg, on="user_id", how="outer")
            .merge(transfer_agg, on="user_id", how="outer")
            .fillna(0)
        )
        flow_metrics["outflow_inflow_ratio"] = (
            flow_metrics["total_outflow"] / flow_metrics["total_inflow"].replace(0, 1)
        )
        flow_metrics["income_spent_pct"] = (
            flow_metrics["total_outflow"] / flow_metrics["total_income"].replace(0, 1)
        ).clip(0, 10)
        flow_metrics.to_csv("data/flow_metrics.csv", index=False)

    print(f"flow_metrics: {len(flow_metrics)} rows")

    # ---- 6) Diversity Features ----
    if os.path.exists("data/diversity_features.csv"):
        diversity_features_df = pd.read_csv("data/diversity_features.csv")
    else:
        tx_sorted = transactions_all.sort_values(["user_id", "date"])
        fa = first_advances.copy()
        fa["underwritten_at"] = _to_dt(fa["underwritten_at"])

        tx_with_adv = tx_sorted.merge(fa[["user_id", "underwritten_at"]], on="user_id", how="inner")
        tx_with_adv["underwritten_at"] = _to_dt(tx_with_adv["underwritten_at"])
        tx_before = tx_with_adv[tx_with_adv["date"] < tx_with_adv["underwritten_at"]]

        diversity_all_time = tx_before.groupby("user_id").agg(
            unique_merchants_all_time=("merchant_name", lambda x: x.dropna().nunique()),
            unique_categories_all_time=("primary_category", "nunique"),
            unique_detailed_categories=("detailed_category", "nunique"),
            first_transaction_date=("date", "min"),
            last_transaction_date=("date", "max"),
            total_transactions_all_time=("id", "count"),
        ).reset_index()

        mc_counts = (
            tx_before.loc[tx_before["merchant_name"].notna(), ["user_id", "merchant_name"]]
            .groupby(["user_id", "merchant_name"]).size().reset_index(name="cnt")
        )
        merchant_concentration = (
            mc_counts.groupby("user_id")["cnt"].apply(lambda c: ((c / c.sum()) ** 2).sum())
            .reset_index(name="merchant_concentration")
        )
        recurring_merchants = (
            mc_counts.loc[mc_counts["cnt"] >= 3]
            .groupby("user_id").size().reset_index(name="recurring_merchants_count")
        )

        diversity_features_df = (
            diversity_all_time.merge(merchant_concentration, on="user_id", how="left")
            .merge(recurring_merchants, on="user_id", how="left")
        )
        diversity_features_df["recurring_merchants_count"] = (
            diversity_features_df["recurring_merchants_count"].fillna(0).astype(int)
        )

        diversity_features_df["time_span_days"] = (
            (_to_dt(diversity_features_df["last_transaction_date"])
             - _to_dt(diversity_features_df["first_transaction_date"]))
            .dt.days.fillna(0).clip(lower=0)
        )
        diversity_features_df["transaction_frequency_all_time"] = (
            diversity_features_df["total_transactions_all_time"]
            / diversity_features_df["time_span_days"].replace(0, 1)
        )

        diversity_features_df = diversity_features_df.merge(
            fa[["user_id", "underwritten_at"]], on="user_id", how="left"
        )
        diversity_features_df["underwritten_at"] = _to_dt(diversity_features_df["underwritten_at"])
        diversity_features_df["days_since_last_transaction"] = (
            (diversity_features_df["underwritten_at"]
             - _to_dt(diversity_features_df["last_transaction_date"]))
            .dt.days.fillna(0)
        )

        # Rolling window features
        for lookback in [30, 60, 90]:
            cutoff_col = f"cutoff_{lookback}d"
            tx_with_adv[cutoff_col] = tx_with_adv["underwritten_at"] - pd.to_timedelta(lookback, unit="D")
            win = tx_with_adv[
                (tx_with_adv["date"] >= tx_with_adv[cutoff_col])
                & (tx_with_adv["date"] < tx_with_adv["underwritten_at"])
            ]
            win_div = (
                win.groupby("user_id").agg(
                    **{
                        f"unique_merchants_{lookback}d": ("merchant_name", lambda x: x.dropna().nunique()),
                        f"unique_categories_{lookback}d": ("primary_category", "nunique"),
                        f"transaction_count_{lookback}d": ("id", "count"),
                    }
                )
                .reset_index()
                .fillna(0)
            )
            diversity_features_df = diversity_features_df.merge(win_div, on="user_id", how="left")

        diversity_features_df = fa[["user_id"]].merge(diversity_features_df, on="user_id", how="left").fillna(0)
        drop_cols = ["first_transaction_date", "last_transaction_date", "underwritten_at"]
        diversity_features_df = diversity_features_df.drop(columns=[c for c in drop_cols if c in diversity_features_df.columns])
        diversity_features_df.to_csv("data/diversity_features.csv", index=False)

    print(f"diversity_features: {len(diversity_features_df)} rows")

    # ---- 7) Join and Target ----
    first_advances = _normalize_id_cols(first_advances, ["request_id"])
    labels_df = _normalize_id_cols(labels_df, ["advance_id"])

    fa_ids = set(first_advances["request_id"].dropna().astype(str))
    lb_ids = set(labels_df["advance_id"].dropna().astype(str))
    inter = len(fa_ids & lb_ids)
    print(f"ID intersection (request_id ↔ advance_id): {inter}")

    labels_join = first_advances.merge(labels_df, left_on="request_id", right_on="advance_id", how="inner")
    print(f"Joined labels: {len(labels_join)} rows")

    # ---- 8) Build supervised dataset ----
    bin_cols = [c for c in labels_df.columns if labels_df[c].dropna().isin([0, 1, True, False]).all()]
    if not bin_cols:
        raise ValueError("No binary target column found in labels.parquet.")
    target_col = bin_cols[0]

    supervised_ids = labels_join[["user_id", target_col]].drop_duplicates()
    features = (
        supervised_ids.merge(spending_pct, on="user_id", how="left")
        .merge(diversity_features_df, on="user_id", how="left")
        .merge(flow_metrics, on="user_id", how="left")
        .fillna(0)
    )

    y = features[target_col].astype(int).values
    X = _ensure_numeric(features.drop(columns=["user_id", target_col]))

    print(f"\nFinal feature matrix: {X.shape}, target: {y.shape}")
    ids = features["user_id"].values  # Track user IDs for export
    return X, y, target_col, ids
