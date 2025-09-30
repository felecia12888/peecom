#!/usr/bin/env python3
"""
Run Validation Checks for Manuscript Claims (A–E)
=================================================

This script verifies the following claims using identical preprocessing,
splits, and model rules across available processed datasets:

A. Cross-dataset average accuracy (+Δ) with 95% CI, paired tests, effect size
B. Robustness / performance drop across source→target transfers
C. Stability score = 100 * (1 - CV) across datasets
D. Feature transferability: (i) performance-transfer ratio; (ii) top-k overlap, with bootstrap CIs
E. Industrial implications: case studies (time-to-detection proxy, energy proxy)

Outputs are saved to: output/validation checks/

Assumptions and notes:
- Uses processed datasets in output/processed_data/<dataset> with X_full.csv and y_full.csv
- Target selection prioritizes binary targets (e.g., 'stable_flag'). Datasets without a usable
  binary target are skipped with a note.
- For cross-dataset transfers, only intersection of columns is used to ensure feature parity.
- MCF baseline = RandomForest on raw processed features (scaled)
- PEECOM = SimplePEECOM (adds physics-like features consistently) on same base features
"""

from __future__ import annotations
import os
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Local imports
from src.models.simple_peecom import SimplePEECOM


OUTPUT_DIR = Path("output") / "validation checks"
PROCESSED_ROOT = Path("output/processed_data")


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def list_processed_datasets() -> List[Path]:
    if not PROCESSED_ROOT.exists():
        return []
    datasets = []
    for d in PROCESSED_ROOT.iterdir():
        if d.is_dir():
            x = d / "X_full.csv"
            y = d / "y_full.csv"
            if x.exists() and y.exists():
                datasets.append(d)
    return datasets


def load_dataset(ds_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    X = pd.read_csv(ds_path / "X_full.csv")
    y = pd.read_csv(ds_path / "y_full.csv")
    return X, y


def pick_binary_target(y: pd.DataFrame) -> Optional[str]:
    # Prefer stable_flag if present
    if "stable_flag" in y.columns:
        return "stable_flag"
    # Fallbacks: look for any column with exactly 2 unique values
    for col in y.columns:
        vals = pd.unique(y[col])
        if len(vals) == 2:
            return col
    return None


def prepare_features_labels(X: pd.DataFrame, y: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, np.ndarray]:
    y_bin = y[target_col].values
    # If labels are not {0,1}, map to binary via first unique as 0, others as 1
    classes = np.unique(y_bin)
    if set(classes) != {0, 1}:
        # Try intelligent mapping for common cases
        mapping = None
        if y_bin.dtype.kind in {"i", "u"} and len(classes) == 2:
            # map smaller -> 0, larger -> 1
            mapping = {classes[0]: 0, classes[1]: 1}
        elif isinstance(classes[0], str) and len(classes) == 2:
            mapping = {classes[0]: 0, classes[1]: 1}
        else:
            # last resort: booleanize by threshold if numeric
            if np.issubdtype(y_bin.dtype, np.number):
                thresh = np.median(y_bin)
                y_bin = (y_bin > thresh).astype(int)
                return X, y_bin
        if mapping is not None:
            y_bin = pd.Series(y_bin).map(mapping).values
    return X, y_bin.astype(int)


def evaluate_with_cv(X: pd.DataFrame, y: np.ndarray, method: str, n_splits: int = 5, seed: int = 42) -> float:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for tr, te in skf.split(X, y):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y[tr], y[te]

        if method == "MCF":
            # Baseline RF on raw features with scaling
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
            clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
            clf.fit(X_tr_s, y_tr)
            pred = clf.predict(X_te_s)
        elif method == "PEECOM":
            clf = SimplePEECOM(n_estimators=200, max_depth=None, random_state=seed)
            clf.fit(X_tr, y_tr)
            pred = clf.predict(X_te)
        else:
            raise ValueError("Unknown method")
        scores.append(accuracy_score(y_te, pred))
    return float(np.mean(scores))


def common_columns(datasets_X: Dict[str, pd.DataFrame]) -> List[str]:
    cols = None
    for _, df in datasets_X.items():
        c = set(df.columns)
        cols = c if cols is None else (cols & c)
    return sorted(cols) if cols else []


def train_on_source_eval_source_target(X_src: pd.DataFrame, y_src: np.ndarray,
                                       X_tgt: pd.DataFrame, y_tgt: np.ndarray,
                                       method: str, seed: int = 42) -> Tuple[float, float]:
    # Train on source (train/val split for source accuracy) and evaluate on full target
    X_tr, X_val, y_tr, y_val = train_test_split(X_src, y_src, test_size=0.2, stratify=y_src, random_state=seed)

    if method == "MCF":
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        X_tgt_s = scaler.transform(X_tgt)
        clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
        clf.fit(X_tr_s, y_tr)
        acc_src = accuracy_score(y_val, clf.predict(X_val_s))
        acc_tgt = accuracy_score(y_tgt, clf.predict(X_tgt_s))
    else:
        clf = SimplePEECOM(n_estimators=200, max_depth=None, random_state=seed)
        clf.fit(X_tr, y_tr)
        acc_src = accuracy_score(y_val, clf.predict(X_val))
        acc_tgt = accuracy_score(y_tgt, clf.predict(X_tgt))
    return acc_src, acc_tgt


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    diff = x - y
    return float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))


def bootstrap_ci(data: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(data)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats.append(np.mean(np.array(data)[idx]))
    low = np.percentile(stats, 100 * (alpha / 2))
    high = np.percentile(stats, 100 * (1 - alpha / 2))
    return float(low), float(high)


def verify_claims():
    ensure_dirs()

    # Discover datasets
    ds_paths = list_processed_datasets()
    if not ds_paths:
        raise RuntimeError("No processed datasets found in output/processed_data.")

    datasets = {}
    skipped = {}
    for d in ds_paths:
        try:
            X, y = load_dataset(d)
            target = pick_binary_target(y)
            if target is None:
                skipped[d.name] = "No binary target available"
                continue
            X_use, y_use = prepare_features_labels(X, y, target)
            # Drop non-numeric columns just in case
            X_use = X_use.select_dtypes(include=[np.number])
            # Remove constant columns
            nunique = X_use.nunique()
            X_use = X_use.loc[:, nunique > 1]
            datasets[d.name] = {
                "X": X_use,
                "y": y_use,
                "target": target,
                "path": str(d)
            }
        except Exception as e:
            skipped[d.name] = f"Load error: {e}"

    # Save discovery report
    (OUTPUT_DIR / "dataset_discovery.json").write_text(json.dumps({
        "used": {k: {"n_features": int(v["X"].shape[1]), "n_samples": int(v["X"].shape[0]), "target": v["target"]} for k, v in datasets.items()},
        "skipped": skipped
    }, indent=2))

    if len(datasets) < 2:
        print("⚠️ Need at least 2 compatible datasets for full cross-dataset analysis. Proceeding with available ones.")

    # A. Cross-dataset average accuracy
    acc_per_dataset_mcf = {}
    acc_per_dataset_peecom = {}
    for name, data in datasets.items():
        acc_mcf = evaluate_with_cv(data["X"], data["y"], method="MCF")
        acc_pe = evaluate_with_cv(data["X"], data["y"], method="PEECOM")
        acc_per_dataset_mcf[name] = acc_mcf
        acc_per_dataset_peecom[name] = acc_pe

    # paired arrays across same datasets
    common_ds = sorted(set(acc_per_dataset_mcf.keys()) & set(acc_per_dataset_peecom.keys()))
    mcf_acc = np.array([acc_per_dataset_mcf[d] for d in common_ds])
    pe_acc = np.array([acc_per_dataset_peecom[d] for d in common_ds])

    def mean_ci(x):
        m = float(np.mean(x))
        s = float(np.std(x, ddof=1))
        n = len(x)
        ci = 1.96 * s / math.sqrt(max(n, 1)) if n > 1 else 0.0
        return m, ci

    mcf_mean, mcf_ci = mean_ci(mcf_acc) if len(mcf_acc) else (float('nan'), float('nan'))
    pe_mean, pe_ci = mean_ci(pe_acc) if len(pe_acc) else (float('nan'), float('nan'))

    # paired tests and effect size
    t_stat, p_val = (np.nan, np.nan)
    w_stat, w_p = (np.nan, np.nan)
    d_eff = np.nan
    if len(common_ds) >= 2:
        t_stat, p_val = ttest_rel(pe_acc, mcf_acc)
        try:
            w_stat, w_p = wilcoxon(pe_acc, mcf_acc)
        except Exception:
            w_stat, w_p = (np.nan, np.nan)
        d_eff = cohens_d(pe_acc, mcf_acc)

    # B. Robustness / performance drop across transfers
    # Build common columns per pair
    drops_mcf = []
    drops_pe = []
    perf_ratio_mcf = []
    perf_ratio_pe = []
    topk_overlap_mcf = []
    topk_overlap_pe = []
    top_k = 20

    names = list(datasets.keys())
    for i, src in enumerate(names):
        for j, tgt in enumerate(names):
            if i == j:
                continue
            Xs, ys = datasets[src]["X"], datasets[src]["y"]
            Xt, yt = datasets[tgt]["X"], datasets[tgt]["y"]
            common = list(set(Xs.columns) & set(Xt.columns))
            if len(common) < 10:
                continue
            Xs_c, Xt_c = Xs[common].copy(), Xt[common].copy()

            # MCF
            src_acc_m, tgt_acc_m = train_on_source_eval_source_target(Xs_c, ys, Xt_c, yt, method="MCF")
            drops_mcf.append(src_acc_m - tgt_acc_m)
            if src_acc_m > 0:
                perf_ratio_mcf.append(tgt_acc_m / src_acc_m)

            # PEECOM
            src_acc_p, tgt_acc_p = train_on_source_eval_source_target(Xs_c, ys, Xt_c, yt, method="PEECOM")
            drops_pe.append(src_acc_p - tgt_acc_p)
            if src_acc_p > 0:
                perf_ratio_pe.append(tgt_acc_p / src_acc_p)

            # D(ii) Top-k feature overlap (train separate models on src and tgt)
            # MCF: RF importances on common features
            rf_src = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
            rf_tgt = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
            scaler_s = StandardScaler().fit(Xs_c)
            scaler_t = StandardScaler().fit(Xt_c)
            rf_src.fit(scaler_s.transform(Xs_c), ys)
            rf_tgt.fit(scaler_t.transform(Xt_c), yt)
            imp_s = rf_src.feature_importances_
            imp_t = rf_tgt.feature_importances_
            order_s = np.argsort(imp_s)[::-1][:top_k]
            order_t = np.argsort(imp_t)[::-1][:top_k]
            top_s = {common[idx] for idx in order_s}
            top_t = {common[idx] for idx in order_t}
            overlap = len(top_s & top_t) / float(top_k)
            topk_overlap_mcf.append(overlap)

            # PEECOM: Use expanded features via SimplePEECOM internal pipeline
            pe_src = SimplePEECOM(n_estimators=200, max_depth=None, random_state=42)
            pe_tgt = SimplePEECOM(n_estimators=200, max_depth=None, random_state=42)
            pe_src.fit(Xs_c, ys)
            pe_tgt.fit(Xt_c, yt)
            # Extract feature names post-expansion by re-creating expansion
            Xs_exp = pe_src._create_physics_features(Xs_c)
            Xt_exp = pe_tgt._create_physics_features(Xt_c)
            # Align common expanded features
            exp_common = list(set(Xs_exp.columns) & set(Xt_exp.columns))
            if len(exp_common) >= top_k:
                # get importances aligned to exp_common order
                # retrain on aligned columns to map importances reliably
                pe_src.fit(Xs_exp[exp_common], ys)
                pe_tgt.fit(Xt_exp[exp_common], yt)
                imp_s2 = pe_src.model.feature_importances_
                imp_t2 = pe_tgt.model.feature_importances_
                order_s2 = np.argsort(imp_s2)[::-1][:top_k]
                order_t2 = np.argsort(imp_t2)[::-1][:top_k]
                top_s2 = {exp_common[idx] for idx in order_s2}
                top_t2 = {exp_common[idx] for idx in order_t2}
                overlap_pe = len(top_s2 & top_t2) / float(top_k)
                topk_overlap_pe.append(overlap_pe)

    drops_mcf = np.array(drops_mcf) if drops_mcf else np.array([])
    drops_pe = np.array(drops_pe) if drops_pe else np.array([])
    perf_ratio_mcf = np.array(perf_ratio_mcf) if perf_ratio_mcf else np.array([])
    perf_ratio_pe = np.array(perf_ratio_pe) if perf_ratio_pe else np.array([])
    topk_overlap_mcf = np.array(topk_overlap_mcf) if topk_overlap_mcf else np.array([])
    topk_overlap_pe = np.array(topk_overlap_pe) if topk_overlap_pe else np.array([])

    # Stats for B and D(i,ii)
    def summarize_arr(arr: np.ndarray, label: str) -> Dict:
        if arr.size == 0:
            return {"n": 0, "mean": None, "ci95": [None, None]}
        mean = float(np.mean(arr))
        lo, hi = bootstrap_ci(arr)
        return {"n": int(arr.size), "mean": mean, "ci95": [lo, hi]}

    drops_summary = {
        "MCF": summarize_arr(drops_mcf, "drops_mcf"),
        "PEECOM": summarize_arr(drops_pe, "drops_pe"),
    }
    perf_transfer_summary = {
        "MCF": summarize_arr(perf_ratio_mcf, "perf_ratio_mcf"),
        "PEECOM": summarize_arr(perf_ratio_pe, "perf_ratio_pe"),
    }
    topk_overlap_summary = {
        "MCF": summarize_arr(topk_overlap_mcf, "topk_overlap_mcf"),
        "PEECOM": summarize_arr(topk_overlap_pe, "topk_overlap_pe"),
    }

    # C. Stability score across datasets
    stability = {}
    if len(common_ds) >= 1:
        def stability_score(x: np.ndarray) -> float:
            m = np.mean(x)
            s = np.std(x, ddof=1) if len(x) > 1 else 0.0
            cv = s / (m + 1e-12)
            return float(100.0 * (1.0 - cv))
        stability = {
            "MCF": stability_score(mcf_acc) if mcf_acc.size else None,
            "PEECOM": stability_score(pe_acc) if pe_acc.size else None,
        }

    # E. Industrial implications (proxy) — time-to-detection and energy proxies
    industrial = {}
    for name, data in datasets.items():
        Xd, yd = data["X"], data["y"]
        # Proxy time index
        time_idx = np.arange(len(yd))
        # Fit PEECOM on first 70%, evaluate probability on full
        X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
            Xd, yd, time_idx, test_size=0.3, stratify=yd, random_state=42)
        pe = SimplePEECOM(n_estimators=200, max_depth=None, random_state=42)
        pe.fit(X_train, y_train)
        # Predict proba on chronological order of test subset
        # Reconstruct order by time index
        order = np.argsort(t_test)
        X_test_ord, y_test_ord = X_test.iloc[order], y_test[order]
        proba = pe.predict_proba(X_test_ord)[:, 1]
        # time-to-detection proxy: first time proba exceeds 0.7 before first true fault in ordered test
        try:
            first_fault_idx = int(np.where(y_test_ord == 1)[0][0])
            first_alarm_idx = int(np.where(proba >= 0.7)[0][0]) if np.any(proba >= 0.7) else None
            lead = None
            if first_alarm_idx is not None:
                lead = first_fault_idx - first_alarm_idx
        except Exception:
            first_fault_idx, first_alarm_idx, lead = None, None, None

        # Energy proxy: correlate EPS1_mean (motor power) if present
        energy_proxy = None
        power_col = None
        for cand in ["EPS1_mean", "EP_mean", "Power_mean"]:
            if cand in Xd.columns:
                power_col = cand
                break
        if power_col:
            try:
                from scipy.stats import spearmanr
                # Evaluate correlation on test subset
                energy_proxy = float(spearmanr(X_test_ord[power_col], proba).correlation)
            except Exception:
                energy_proxy = None

        industrial[name] = {
            "time_to_detection_proxy": {
                "first_fault_idx": first_fault_idx,
                "first_alarm_idx": first_alarm_idx,
                "lead_samples": lead
            },
            "energy_proxy_correlation": energy_proxy,
        }

    # Compose final report
    report = {
        "datasets_used": common_ds,
        "dataset_details": {k: {"n": int(v["X"].shape[0]), "p": int(v["X"].shape[1]), "target": v["target"]} for k, v in datasets.items()},
        "skipped_datasets": skipped,
        "A_cross_dataset_accuracy": {
            "MCF": {"mean": mcf_mean, "ci95": mcf_ci},
            "PEECOM": {"mean": pe_mean, "ci95": pe_ci},
            "paired_t": {"t": None if math.isnan(t_stat) else float(t_stat), "p": None if math.isnan(p_val) else float(p_val)},
            "wilcoxon": {"stat": None if math.isnan(w_stat) else float(w_stat), "p": None if math.isnan(w_p) else float(w_p)},
            "effect_size_cohens_d": None if math.isnan(d_eff) else float(d_eff),
            "per_dataset": {d: {"MCF": float(acc_per_dataset_mcf[d]), "PEECOM": float(acc_per_dataset_peecom[d])} for d in common_ds}
        },
        "B_robustness_performance_drop": drops_summary,
        "C_stability_score": stability,
        "D_transferability": {
            "performance_transfer_ratio": perf_transfer_summary,
            "topk_feature_overlap": topk_overlap_summary,
            "top_k": top_k
        },
        "E_industrial_implications": industrial,
    }

    # Save JSON and CSV summaries
    (OUTPUT_DIR / "claims_verification_report.json").write_text(json.dumps(report, indent=2))

    # Also provide compact CSVs for A, B, D
    if common_ds:
        pd.DataFrame({
            "dataset": common_ds,
            "MCF_accuracy": [acc_per_dataset_mcf[d] for d in common_ds],
            "PEECOM_accuracy": [acc_per_dataset_peecom[d] for d in common_ds],
        }).to_csv(OUTPUT_DIR / "A_cross_dataset_accuracy.csv", index=False)

    if drops_mcf.size or drops_pe.size:
        pd.DataFrame({
            "drop_MCF": drops_mcf,
            "drop_PEECOM": drops_pe,
        }).to_csv(OUTPUT_DIR / "B_performance_drops.csv", index=False)

    if perf_ratio_mcf.size or perf_ratio_pe.size:
        pd.DataFrame({
            "ratio_MCF": perf_ratio_mcf,
            "ratio_PEECOM": perf_ratio_pe,
        }).to_csv(OUTPUT_DIR / "D_performance_transfer_ratio.csv", index=False)

    if topk_overlap_mcf.size or topk_overlap_pe.size:
        pd.DataFrame({
            "topk_overlap_MCF": topk_overlap_mcf,
            "topk_overlap_PEECOM": topk_overlap_pe,
        }).to_csv(OUTPUT_DIR / "D_topk_feature_overlap.csv", index=False)

    print("✅ Claims verification complete. See:")
    print(f" - {OUTPUT_DIR / 'claims_verification_report.json'}")
    print(f" - {OUTPUT_DIR}")


if __name__ == "__main__":
    verify_claims()
