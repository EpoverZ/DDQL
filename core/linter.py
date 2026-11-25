# =============================#
#       Structured Linter      #
# =============================#

import io
import os
import re
import contextlib
from typing import Union, Optional, Dict, Any, List
import pandas as pd

from core.loader import load_dataset, detect_file_format
from core.summary import get_dataset_summary
from core.checks import (
    _sanitize_unhashable,
    check_missing_values,
    check_duplicates,
    check_constant_columns,
    validate_target_column,
    class_imbalance,
    leakage_checks,
    bias_signals,
    dataset_health,
    trainability_checks,
)

VERSION = "0.8.0"


def warning_to_finding(warn_str: str, target: Optional[str] = None) -> Dict[str, Any]:
    """
    Parses a warning string into a structured finding object.
    Categories: data_quality, target, leakage, bias, trainability.
    Levels: warn, info.
    """
    level = "warn" if "[WARN]" in warn_str or "[CRITICAL]" in warn_str else "info"

    # 1. Missing values
    m = re.search(r"Column '(.+?)' has (.+?)% missing values", warn_str)
    if m:
        return {
            "level": level,
            "category": "data_quality",
            "code": "missing_values_threshold_exceeded",
            "message": warn_str,
            "columns": [m.group(1)],
        }

    # 2. Duplicate rows
    if "duplicate rows" in warn_str:
        return {
            "level": level,
            "category": "data_quality",
            "code": "duplicate_rows_detected",
            "message": warn_str,
            "columns": [],
        }

    # 3. Constant column
    m = re.search(r"Column '(.+?)' is constant", warn_str)
    if m:
        return {
            "level": level,
            "category": "data_quality",
            "code": "constant_column",
            "message": warn_str,
            "columns": [m.group(1)],
        }

    # 4. Label dominance
    m = re.search(r"Label '(.+?)' dominates dataset", warn_str)
    if m:
        return {
            "level": level,
            "category": "target",
            "code": "label_dominance",
            "message": warn_str,
            "columns": [target] if target else [],
        }

    # 5. Rare class
    m = re.search(r"Label '(.+?)' has only (\d+) samples", warn_str)
    if m:
        return {
            "level": level,
            "category": "target",
            "code": "rare_class",
            "message": warn_str,
            "columns": [target] if target else [],
        }

    # 6. Correlation leakage
    m = re.search(r"Feature '(.+?)' highly correlated with target", warn_str)
    if m:
        cols = [m.group(1)]
        if target:
            cols.append(target)
        return {
            "level": level,
            "category": "leakage",
            "code": "high_target_correlation",
            "message": warn_str,
            "columns": cols,
        }

    # 7. ID-like column
    m = re.search(r"Feature '(.+?)' appears ID-like", warn_str)
    if m:
        return {
            "level": level,
            "category": "leakage",
            "code": "id_like_column",
            "message": warn_str,
            "columns": [m.group(1)],
        }

    # 8. Strong leakage risk
    m = re.search(r"Feature '(.+?)' shows strong leakage risk", warn_str)
    if m:
        cols = [m.group(1)]
        if target:
            cols.append(target)
        return {
            "level": level,
            "category": "leakage",
            "code": "strong_leakage_risk",
            "message": warn_str,
            "columns": cols,
        }

    # 9. Bias under-represented group
    m = re.search(r"Group '(.+?)' in '(.+?)' under-represented", warn_str)
    if m:
        return {
            "level": level,
            "category": "bias",
            "code": "group_underrepresented",
            "message": warn_str,
            "columns": [m.group(2)],
        }

    # 10. Small dataset
    if "Dataset is very small" in warn_str:
        return {
            "level": level,
            "category": "data_quality",
            "code": "small_dataset",
            "message": warn_str,
            "columns": [],
        }

    # 11. More features than samples
    if "More features than samples" in warn_str:
        return {
            "level": level,
            "category": "data_quality",
            "code": "features_exceed_samples",
            "message": warn_str,
            "columns": [],
        }

    # 12. Missing values trainability
    if "Missing values must be handled before training" in warn_str:
        return {
            "level": level,
            "category": "trainability",
            "code": "missing_values_unhandled",
            "message": warn_str,
            "columns": [],
        }

    # 13. No numeric features
    if "No numeric features for ML models" in warn_str:
        return {
            "level": level,
            "category": "trainability",
            "code": "no_numeric_features",
            "message": warn_str,
            "columns": [],
        }

    # Generic fallback
    return {
        "level": level,
        "category": "data_quality",
        "code": "general_warning",
        "message": warn_str,
        "columns": [],
    }


def lint_dataset(
    df_or_path: Union[str, pd.DataFrame],
    target: Optional[str] = None,
    missing_threshold: float = 30.0,
    dup_threshold: float = 5.0,
    dominance_threshold: float = 90.0,
    rare_count_threshold: int = 20,
    corr_threshold: float = 0.9,
    id_ratio: float = 0.95,
    min_group_pct: float = 10.0,
    capture_stdout: bool = True,
) -> Dict[str, Any]:
    """
    Executes the tabular dataset linting rules and returns a structured dictionary.
    Includes detected 'file_format' in summary, structured 'findings', and stats.
    """
    stdout_capture = io.StringIO()
    cm = contextlib.redirect_stdout(stdout_capture) if capture_stdout else contextlib.nullcontext()

    with cm:
        if isinstance(df_or_path, pd.DataFrame):
            df = df_or_path.copy()
            file_format = "dataframe"
        else:
            if not os.path.exists(df_or_path):
                raise FileNotFoundError(f"Dataset file not found: '{df_or_path}'")
            file_format = detect_file_format(df_or_path)
            df = load_dataset(df_or_path)

        summary = get_dataset_summary(df, file_format=file_format)
        warnings: List[str] = []
        errors: List[str] = []

        # Sanitize dataframe for unhashable types (nested lists/dicts)
        safe_df = _sanitize_unhashable(df)

        # Missing values stats & warnings
        missing_counts = df.isna().sum()
        missing_stats = {
            str(col): {
                "count": int(count),
                "percentage": round(float((count / len(df)) * 100), 2) if len(df) > 0 else 0.0,
            }
            for col, count in missing_counts.items()
            if count > 0
        }
        warnings.extend(check_missing_values(df, threshold=missing_threshold))

        # Duplicate rows stats & warnings (safe against unhashables)
        dup_count = int(safe_df.duplicated().sum())
        dup_pct = round(float((dup_count / len(safe_df)) * 100), 2) if len(safe_df) > 0 else 0.0
        warnings.extend(check_duplicates(df, threshold=dup_threshold))

        # Constant columns stats & warnings (safe against unhashables)
        constant_cols = [str(col) for col, unique in safe_df.nunique().items() if unique == 1]
        warnings.extend(check_constant_columns(df))

        # Target analysis
        target_info = None
        if target is not None and str(target).strip():
            target_str = str(target).strip()
            if target_str not in df.columns:
                errors.append(f"Target column '{target_str}' not found in dataset columns: {list(df.columns)}")
            else:
                target_series = safe_df[target_str]
                counts = target_series.value_counts()
                percents = target_series.value_counts(normalize=True) * 100
                target_info = {
                    "column": target_str,
                    "class_frequencies": {str(k): int(v) for k, v in counts.items()},
                    "class_percentages": {str(k): round(float(v), 2) for k, v in percents.items()},
                }

                warnings.extend(
                    class_imbalance(
                        df,
                        target_str,
                        dominance_threshold=dominance_threshold,
                        rare_count_threshold=rare_count_threshold,
                    )
                )
                warnings.extend(
                    leakage_checks(
                        df,
                        target_str,
                        corr_threshold=corr_threshold,
                        id_ratio=id_ratio,
                    )
                )
                warnings.extend(
                    bias_signals(
                        df,
                        target_str,
                        min_group_pct=min_group_pct,
                    )
                )

        # Dataset health & trainability
        warnings.extend(dataset_health(df))
        warnings.extend(trainability_checks(df))

        # Generate structured findings parallel to warnings
        effective_target = str(target).strip() if target and str(target).strip() else None
        findings = [warning_to_finding(w, target=effective_target) for w in warnings]

    return {
        "status": "error" if errors else "ok",
        "version": VERSION,
        "summary": summary,
        "warnings": warnings,
        "findings": findings,
        "errors": errors,
        "stats": {
            "missing_values": missing_stats,
            "duplicate_rows": {
                "count": dup_count,
                "percentage": dup_pct,
            },
            "constant_columns": constant_cols,
            "target_analysis": target_info,
        },
    }
