#####################
## ML-DATASET-LINT ##
#####################

import os
import sys
import json
import argparse
from typing import Optional, List

from core.loader import load_dataset
from core.summary import print_dataset_summary
from core.checks import (
    check_missing_values,
    check_duplicates,
    check_constant_columns,
    validate_target_column,
    extract_column,
    class_frequencies,
    class_percents,
    class_imbalance,
    leakage_checks,
    bias_signals,
    dataset_health,
    trainability_checks,
)
from core.report import print_warnings
from core.linter import lint_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dqdlz",
        description="DQDLz: Fast, rule-based ML dataset linter for tabular data quality, target imbalance, leakage, and trainability."
    )
    parser.add_argument(
        "file",
        nargs="?",
        default=None,
        help="Path to dataset file (.csv, .tsv, .txt, .xlsx, .xls, .json, .jsonl, .ndjson, .parquet, .feather). Falls back to interactive mode if omitted."
    )
    parser.add_argument(
        "-t", "--target",
        dest="target",
        default=None,
        help="Target column name for supervised ML checks (leakage, class imbalance, bias signals)."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output structured JSON instead of human-readable terminal text (ideal for CI/CD and scripting)."
    )
    parser.add_argument(
        "--missing-threshold",
        type=float,
        default=30.0,
        help="Missing values percentage threshold to trigger warning (default: 30.0)."
    )
    parser.add_argument(
        "--dup-threshold",
        type=float,
        default=5.0,
        help="Duplicate rows percentage threshold to trigger warning (default: 5.0)."
    )
    parser.add_argument(
        "--dominance-threshold",
        type=float,
        default=90.0,
        help="Class dominance percentage threshold to trigger warning (default: 90.0)."
    )
    parser.add_argument(
        "--rare-count-threshold",
        type=int,
        default=20,
        help="Minimum sample count per class before triggering rare class warning (default: 20)."
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.9,
        help="Feature correlation with target threshold to trigger leakage warning (default: 0.9)."
    )
    parser.add_argument(
        "--id-ratio",
        type=float,
        default=0.95,
        help="Unique values ratio threshold to flag ID-like feature (default: 0.95)."
    )
    parser.add_argument(
        "--min-group-pct",
        type=float,
        default=10.0,
        help="Minimum group percentage in sensitive columns before triggering bias warning (default: 10.0)."
    )
    return parser


def run_cli(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    path = args.file
    target = args.target

    # Interactive prompt fallback if no file passed
    if not path:
        try:
            path = input("ENTER DATASET FILE PATH (.csv, .tsv, .xlsx, .json, .parquet, .feather): ").strip()
        except (EOFError, KeyboardInterrupt):
            path = ""

        if not path:
            print("[ERROR] No dataset path provided.")
            return 1

        if target is None:
            try:
                target_in = input("ENTER TARGET COLUMN (OR LEAVE BLANK): ").strip()
                target = target_in if target_in else None
            except (EOFError, KeyboardInterrupt):
                target = None
    else:
        path = path.strip()

    # JSON output mode
    if args.json_output:
        try:
            result = lint_dataset(
                path,
                target=target,
                missing_threshold=args.missing_threshold,
                dup_threshold=args.dup_threshold,
                dominance_threshold=args.dominance_threshold,
                rare_count_threshold=args.rare_count_threshold,
                corr_threshold=args.corr_threshold,
                id_ratio=args.id_ratio,
                min_group_pct=args.min_group_pct,
                capture_stdout=True,
            )
            print(json.dumps(result, indent=2))
            return 1 if result.get("status") == "error" else 0
        except Exception as e:
            err_dict = {
                "status": "error",
                "errors": [str(e)],
                "warnings": [],
                "findings": [],
            }
            print(json.dumps(err_dict, indent=2))
            return 1

    # Human-readable terminal output mode
    try:
        df, file_format = load_dataset(path, return_format=True)
    except Exception as e:
        print(f"[ERROR] {e}")
        return 1

    print_dataset_summary(df, file_format=file_format)

    warnings = []

    # -----------------------------
    # v0.1 — Basic dataset checks
    # -----------------------------
    warnings.extend(check_missing_values(df, threshold=args.missing_threshold))
    warnings.extend(check_duplicates(df, threshold=args.dup_threshold))
    warnings.extend(check_constant_columns(df))

    print("=" * 55)

    # -----------------------------
    # v0.2 — Target analysis
    # -----------------------------
    if target and validate_target_column(df, target):
        X, y = extract_column(df, target)

        print("\nTarget preview:")
        print(y.head())

        class_frequencies(df, target)
        class_percents(df, target)

        warnings.extend(
            class_imbalance(
                df,
                target,
                dominance_threshold=args.dominance_threshold,
                rare_count_threshold=args.rare_count_threshold,
            )
        )

        print("=" * 55)

        # -----------------------------
        # v0.3 — Leakage checks
        # -----------------------------
        warnings.extend(
            leakage_checks(
                df,
                target,
                corr_threshold=args.corr_threshold,
                id_ratio=args.id_ratio,
            )
        )

        # -----------------------------
        # v0.4 — Bias signals
        # -----------------------------
        warnings.extend(
            bias_signals(
                df,
                target,
                min_group_pct=args.min_group_pct,
            )
        )
    elif not target:
        print("[INFO] No target provided -- skipping target-based checks")

    # -----------------------------
    # v0.5 — Dataset health
    # -----------------------------
    warnings.extend(dataset_health(df))

    # -----------------------------
    # v0.6 — Trainability
    # -----------------------------
    warnings.extend(trainability_checks(df))

    print("\n================ LINT WARNINGS ================")
    print_warnings(warnings)

    return 0


def main(argv: Optional[List[str]] = None) -> None:
    code = run_cli(argv)
    if code != 0:
        sys.exit(code)


if __name__ == "__main__":
    main()
