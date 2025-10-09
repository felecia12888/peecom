"""Utility functions for detecting and remediating data leakage in processed datasets.

This module provides a focused implementation for analysing associations between
features and targets in the processed PEECOM datasets.  It inspects strong
linear relationships, mutual information signals, and near-identical patterns to
surface candidate leakage features.  The resulting report is intentionally
light-weight and JSON serialisable so that automation scripts can persist and
aggregate the findings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder


@dataclass
class LeakageIssue:
    """Container describing a potential leakage finding."""

    feature: str
    issue: str
    severity: str
    target: Optional[str] = None
    score: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "feature": self.feature,
            "issue": self.issue,
            "severity": self.severity,
        }
        if self.target is not None:
            payload["target"] = self.target
        if self.score is not None:
            payload["score"] = self.score
        if self.details:
            payload["details"] = self.details
        return payload


@dataclass
class LeakageReport:
    """Structured leakage report for a single dataset."""

    dataset_name: str
    samples: int
    features: int
    targets: int
    feature_columns: List[str]
    target_columns: List[str]
    issues: List[LeakageIssue] = field(default_factory=list)
    correlation_threshold: float = 0.95
    mi_threshold: float = 0.95
    match_threshold: float = 0.99
    generated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        issues_as_dicts = [issue.to_dict() for issue in self.issues]
        severity_counts = {
            "critical": sum(1 for issue in self.issues if issue.severity == "critical"),
            "high": sum(1 for issue in self.issues if issue.severity == "high"),
            "medium": sum(1 for issue in self.issues if issue.severity == "medium"),
        }
        removal_candidates = {
            issue.feature
            for issue in self.issues
            if issue.issue != "high_mutual_information" or issue.severity == "critical"
        }
        return {
            "dataset_name": self.dataset_name,
            "analysis_timestamp": self.generated_at.isoformat(),
            "data_info": {
                "samples": self.samples,
                "features": self.features,
                "targets": self.targets,
                "feature_columns": self.feature_columns,
                "target_columns": self.target_columns,
            },
            "parameters": {
                "correlation_threshold": self.correlation_threshold,
                "mi_threshold": self.mi_threshold,
                "match_threshold": self.match_threshold,
            },
            "leakage_issues": {
                "total_issues": len(self.issues),
                "critical_issues": severity_counts["critical"],
                "high_issues": severity_counts["high"],
                "medium_issues": severity_counts["medium"],
                "all_issues": issues_as_dicts,
                "features_to_remove": sorted(removal_candidates),
            },
        }


def _encode_target(target: pd.Series) -> np.ndarray:
    encoder = LabelEncoder()
    return encoder.fit_transform(target.astype(str))


def _safe_corr(features: pd.DataFrame, target: pd.Series) -> pd.Series:
    joined = pd.concat(
        [features, target.rename("__target__")], axis=1).dropna()
    if joined.empty:
        return pd.Series(dtype=float)
    correlations = joined.corr()["__target__"].drop("__target__")
    return correlations.fillna(0.0)


def _detect_matching_series(feature: pd.Series, target: pd.Series, threshold: float) -> Optional[float]:
    """Return match ratio if the feature nearly replicates the target."""
    if feature.isna().all() or target.isna().all():
        return None
    aligned = pd.concat([feature, target], axis=1).dropna()
    if aligned.empty:
        return None

    if feature.dtype.kind in "bifc" and target.dtype.kind in "bifc":
        # Numeric comparison with tolerance
        equal_ratio = np.mean(np.isclose(
            aligned.iloc[:, 0], aligned.iloc[:, 1], rtol=0.0, atol=1e-6))
    else:
        equal_ratio = np.mean(aligned.iloc[:, 0].astype(
            str) == aligned.iloc[:, 1].astype(str))

    return equal_ratio if equal_ratio >= threshold else None


def detect_leakage(
    X: pd.DataFrame,
    y: pd.DataFrame,
    dataset_name: str,
    correlation_threshold: float = 0.95,
    mi_threshold: float = 0.5,
    match_threshold: float = 0.99,
) -> LeakageReport:
    """Analyse leakage signals between features and targets.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.DataFrame
        Target matrix (each column treated independently).
    dataset_name : str
        Identifier used in the returned report.
    correlation_threshold : float, optional
        Absolute Pearson correlation threshold that triggers a leakage flag.
    mi_threshold : float, optional
        Mutual information threshold for discrete target relationships.
    match_threshold : float, optional
        Ratio of identical values required to treat a feature as a direct copy
        of a target.
    """

    issues: List[LeakageIssue] = []

    for target_name in y.columns:
        target_series = y[target_name]
        if target_series.nunique(dropna=True) <= 1:
            continue

        correlations = _safe_corr(X, target_series)
        high_corr = correlations[correlations.abs() >= correlation_threshold]
        for feature_name, corr_value in high_corr.items():
            severity = "critical" if abs(
                corr_value) >= correlation_threshold + 0.03 else "high"
            issues.append(
                LeakageIssue(
                    feature=feature_name,
                    issue="high_correlation",
                    severity=severity,
                    target=target_name,
                    score=float(corr_value),
                )
            )

        match_candidates = []
        for feature_name in X.columns:
            match_ratio = _detect_matching_series(
                X[feature_name], target_series, match_threshold)
            if match_ratio is not None:
                match_candidates.append((feature_name, match_ratio))

        for feature_name, ratio in match_candidates:
            issues.append(
                LeakageIssue(
                    feature=feature_name,
                    issue="near_duplicate_target",
                    severity="critical",
                    target=target_name,
                    score=float(ratio),
                )
            )

        # Mutual information for discrete targets (categorical or low cardinality)
        n_unique = target_series.nunique(dropna=True)
        if n_unique <= 20:
            encoded_target = _encode_target(target_series)
            try:
                mi_values = mutual_info_classif(
                    X.fillna(0.0), encoded_target, random_state=42)
            except Exception:
                # Some sklearn versions require finite variance; fall back to standardised copy
                safe_X = X.fillna(0.0)
                safe_X = (safe_X - safe_X.mean()) / (safe_X.std(ddof=0) + 1e-9)
                mi_values = mutual_info_classif(
                    safe_X.fillna(0.0), encoded_target, random_state=42)

            mi_series = pd.Series(mi_values, index=X.columns)
            max_possible_mi = float(np.log(max(n_unique, 2)))
            normalised_mi = mi_series / \
                (max_possible_mi if max_possible_mi > 0 else 1.0)
            strong_mi = normalised_mi[normalised_mi >= mi_threshold]
            for feature_name, mi_ratio in strong_mi.items():
                severity = "high" if mi_ratio < mi_threshold * 1.1 else "critical"
                issues.append(
                    LeakageIssue(
                        feature=feature_name,
                        issue="high_mutual_information",
                        severity=severity,
                        target=target_name,
                        score=float(mi_ratio),
                        details={
                            "mutual_information": float(mi_series[feature_name]),
                            "normalised_mutual_information": float(mi_ratio),
                        },
                    )
                )

    # Merge duplicate entries for the same feature/issue pair by keeping the strongest
    merged: Dict[tuple, LeakageIssue] = {}
    for issue in issues:
        key = (issue.feature, issue.issue, issue.target)
        existing = merged.get(key)
        if not existing or (issue.score or 0) > (existing.score or 0):
            merged[key] = issue

    consolidated_issues = list(merged.values())

    report = LeakageReport(
        dataset_name=dataset_name,
        samples=len(X),
        features=X.shape[1],
        targets=y.shape[1],
        feature_columns=list(X.columns),
        target_columns=list(y.columns),
        issues=consolidated_issues,
        correlation_threshold=correlation_threshold,
        mi_threshold=mi_threshold,
        match_threshold=match_threshold,
    )
    return report


def summarise_reports(reports: Iterable[LeakageReport]) -> Dict[str, Any]:
    reports = list(reports)
    datasets = {report.dataset_name: report.to_dict() for report in reports}
    datasets_with_issues = sum(
        1 for report in reports if report.to_dict()["leakage_issues"]["total_issues"] > 0
    )
    return {
        "analysis_timestamp": datetime.utcnow().isoformat(),
        "total_datasets": len(reports),
        "datasets_analyzed": len(reports),
        "datasets_with_issues": datasets_with_issues,
        "datasets": datasets,
    }
