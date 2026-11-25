#####################
## ML-DATASET-LINT ##
#####################

import sys
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


def main():
    # -----------------------------
    # Load dataset (Universal format support: CSV, TSV, Excel, JSON, Parquet, Feather)
    # -----------------------------
    if len(sys.argv) > 1:
        path = sys.argv[1].strip()
    else:
        try:
            path = input("ENTER DATASET FILE PATH (.csv, .tsv, .xlsx, .json, .parquet, .feather): ").strip()
        except EOFError:
            path = ""

    if not path:
        print("[ERROR] No dataset path provided.")
        return

    df, file_format = load_dataset(path, return_format=True)

    print_dataset_summary(df, file_format=file_format)

    warnings = []

    # -----------------------------
    # v0.1 — Basic dataset checks
    # -----------------------------
    warnings.extend(check_missing_values(df))
    warnings.extend(check_duplicates(df))
    warnings.extend(check_constant_columns(df))

    print("=" * 55)

    # -----------------------------
    # v0.2 — Target analysis
    # -----------------------------
    if len(sys.argv) > 2:
        target = sys.argv[2].strip()
    else:
        try:
            target = input("ENTER TARGET COLUMN (OR LEAVE BLANK): ").strip()
        except EOFError:
            target = ""

    if validate_target_column(df, target):
        X, y = extract_column(df, target)

        print("\nTarget preview:")
        print(y.head())

        class_frequencies(df, target)
        class_percents(df, target)

        warnings.extend(
            class_imbalance(
                df,
                target,
                dominance_threshold=90,
                rare_count_threshold=20,
            )
        )

        print("=" * 55)

        # -----------------------------
        # v0.3 — Leakage checks
        # -----------------------------
        warnings.extend(leakage_checks(df, target))

        # -----------------------------
        # v0.4 — Bias signals
        # -----------------------------
        warnings.extend(bias_signals(df, target))

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


if __name__ == "__main__":
    main()
