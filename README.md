# DQDLz (`ml-dataset-lint`)

**DQDLz** (DataQuality & DataLintz) is a fast, lightweight, rule-based terminal CLI and library for linting machine learning datasets before training.

Just as a code linter flags bugs, type errors, and anti-patterns before compilation, DQDLz scans tabular data to detect data quality anomalies, severe class imbalances, feature leakage, demographic representation bias, and structural trainability issues **before** you feed data to your models.

---

## Key Features

- **10 Supported Formats**: Automatically detects and loads tabular data across formats without manual conversion.
- **Rule-Based & Deterministic**: Zero black-box guesswork — every check uses transparent, customizable mathematical thresholds.
- **Leakage & Bias Detection**: Identifies features suspiciously correlated with targets, ID-like features, and under-represented demographic groups.
- **Dual CLI Modes**: Use interactive prompts or direct command-line flags.
- **Scripting & CI/CD Ready**: Output clean, structured JSON with `--json` for integration into automated pipelines.
- **Unhashable Safe**: Gracefully handles nested structures (lists, dictionaries, JSON) without crashing uniqueness or duplicate row checks.
- **Zero Cloud Overhead**: Runs 100% locally on your machine with no external network calls, no telemetry, and no artificial file size caps.

---

## Supported Dataset Formats

DQDLz supports all major tabular and semi-structured file formats:

| Format | File Extensions | Notes |
| :--- | :--- | :--- |
| **CSV** | `.csv` | Automatic delimiter sniffing and encoding fallbacks (`utf-8`, `utf-8-sig`, `latin1`, `cp1252`). |
| **TSV / Plain Text** | `.tsv`, `.txt` | Tab-delimited tabular datasets with encoding fallbacks. |
| **Excel** | `.xlsx`, `.xls` | Single and multi-column spreadsheets loaded via `openpyxl`. |
| **JSON** | `.json` | Standard JSON records or wrapped objects normalized via `pd.json_normalize`. |
| **Newline JSON** | `.jsonl`, `.ndjson` | High-throughput line-delimited JSON streams. |
| **Apache Parquet** | `.parquet` | Columnar binary format powered by `pyarrow`. |
| **Apache Arrow Feather** | `.feather` | Ultra-fast memory-mapped binary storage powered by `pyarrow`. |

---

## Installation

Clone the repository and install the lightweight dependencies:

```bash
git clone https://github.com/your-username/ml-dataset-lint.git
cd ml-dataset-lint
pip install -r requirements.txt
```

### Dependencies
- `python>=3.9`
- `numpy>=1.24.0`
- `pandas>=2.0.0`
- `openpyxl>=3.1.0`
- `pyarrow>=14.0.0`

*(Optional for running test suite)*:
```bash
pip install pytest
```

---

## CLI Usage

### 1. Interactive Mode
Run without arguments to launch the interactive prompt:
```bash
python main.py
```
```text
ENTER DATASET FILE PATH (.csv, .tsv, .xlsx, .json, .parquet, .feather): data/churn.parquet
ENTER TARGET COLUMN (OR LEAVE BLANK): churned
```

### 2. Flag-Based Mode

#### Unsupervised Linting (No Target Column)
```bash
python main.py data.csv
```

#### Supervised Linting (With Target Column)
Specify `--target` (or `-t`) to activate target distribution, label dominance, leakage, and sensitive feature checks:
```bash
python main.py data.parquet --target label
```

#### JSON Output Mode (for CI/CD & Scripting)
Add `--json` to pipe clean structured JSON directly to tools like `jq`, Python, or CI/CD runners:
```bash
python main.py data.xlsx --target label --json
```

Example JSON structure:
```json
{
  "status": "ok",
  "version": "0.8.0",
  "summary": {
    "file_format": "excel",
    "rows": 1000,
    "columns": 12,
    "column_types": { ... }
  },
  "warnings": [
    "[WARN] Column 'age' has 34.50% missing values",
    "[WARN] Feature 'account_id' appears ID-like"
  ],
  "findings": [
    {
      "level": "warn",
      "category": "data_quality",
      "code": "missing_values_threshold_exceeded",
      "message": "[WARN] Column 'age' has 34.50% missing values",
      "columns": ["age"]
    }
  ],
  "stats": {
    "missing_values": { "age": { "count": 345, "percentage": 34.5 } },
    "duplicate_rows": { "count": 0, "percentage": 0.0 },
    "constant_columns": [],
    "target_analysis": { ... }
  }
}
```

### 3. Custom Threshold Flags

Tune any linting sensitivity threshold directly via command-line arguments:

```bash
python main.py dataset.csv --target is_fraud \
  --missing-threshold 20.0 \
  --dup-threshold 2.0 \
  --corr-threshold 0.85 \
  --dominance-threshold 80.0 \
  --rare-count-threshold 50
```

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--target`, `-t` | `None` | Target column name for supervised analysis. |
| `--json` | `False` | Output structured JSON instead of human-readable text. |
| `--missing-threshold` | `30.0` | Max missing value percentage before flagging a column (`%`). |
| `--dup-threshold` | `5.0` | Max duplicate row percentage before flagging dataset (`%`). |
| `--dominance-threshold`| `90.0` | Max class dominance percentage before flagging imbalance (`%`). |
| `--rare-count-threshold`| `20` | Minimum sample count required per class before flagging rare classes. |
| `--corr-threshold` | `0.9` | Absolute Pearson correlation with target indicating leakage (`0.0 - 1.0`). |
| `--id-ratio` | `0.95` | Unique value ratio indicating an ID-like column (`0.0 - 1.0`). |
| `--min-group-pct` | `10.0` | Minimum demographic subgroup share before flagging bias (`%`). |

View all flags anytime with:
```bash
python main.py --help
```

---

## What Checks Are Performed?

DQDLz groups rules into five core categories:

1. **Data Quality (`data_quality`)**:
   - Missing value percentage per column.
   - Exact duplicate row percentages (unhashable-safe).
   - Zero-variance constant columns.
   - Small dataset warnings (`< 100` rows).
   - High-dimensionality / curse of dimensionality (`cols > rows`).

2. **Target & Imbalance (`target`)**:
   - Class dominance detection (`> 90%` of samples in single class).
   - Rare class representation (`< 20` samples in minority class).

3. **Data Leakage (`leakage`)**:
   - Extreme feature-to-target correlation (`|r| > 0.90`).
   - High-cardinality ID-like columns (`nunique / rows > 0.95`).
   - Critical leakage risk (columns that are both ID-like and highly correlated).

4. **Demographic Bias Signals (`bias`)**:
   - Under-represented groups in sensitive columns (`gender`, `sex`, `race`, `age`, `region`, `income`).

5. **Trainability & Health (`trainability`)**:
   - Verification of presence of numeric features for ML models.
   - Unhandled missing value presence blocking common estimators.

For full mathematical definitions, codes, and rationale, see [docs/CHECKS.md](docs/CHECKS.md).

---

## Python API Usage

You can also import and use DQDLz directly within your Python scripts or notebooks:

```python
from core.linter import lint_dataset

# Lint from a file path or an in-memory pandas DataFrame
report = lint_dataset("data.parquet", target="churn", missing_threshold=25.0)

if report["findings"]:
    print(f"Detected {len(report['findings'])} issues:")
    for finding in report["findings"]:
        print(f" - [{finding['level'].upper()}] {finding['category']}: {finding['message']}")
```

---

## Running Tests

Execute the comprehensive test suite with `pytest`:

```bash
pytest -v tests/
```

The test suite validates:
- CLI argument parsing and threshold flags.
- Terminal output formatting and `--json` schemas.
- Ingestion across CSV, Excel, JSON, and Parquet.
- Error handling on missing files, corrupt data, and invalid column names.
- Interactive mode fallbacks.

---

## Project Structure

```text
ml-dataset-lint/
|-- .github/
|   |-- workflows/
|       |-- ci.yml           # Automated CLI test suite on push/PR
|-- core/
|   |-- loader.py            # 10-format universal dataset loader
|   |-- summary.py           # Dataset schema overview & shape summaries
|   |-- checks.py            # Core linting rules & unhashable sanitization
|   |-- report.py            # Console output formatting
|   |-- linter.py            # Structured linter engine & finding mapper
|-- docs/
|   |-- CHECKS.md            # Detailed catalog of all checks, formulas, and thresholds
|-- tests/
|   |-- test_cli.py          # Pytest CLI and format test suite
|-- main.py                  # CLI entry point (interactive & flag-driven)
|-- requirements.txt         # Lightweight Python dependencies
|-- README.md                # Project documentation
|-- LICENSE                  # GNU General Public License v3.0
```

---

## License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**. See the [LICENSE](LICENSE) file for details.
