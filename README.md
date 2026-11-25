# DQDLz (DataQuality & DataLintz)

DQDLz is a lightweight, rule-based dataset linting engine and REST API designed to detect data quality, target distribution, feature leakage, demographic bias, and structural trainability risks **before** training machine learning models.

It functions like a code linter for datasets — flagging missing values, row duplication, constant columns, severe class imbalance, high target correlation leakage, and few-shot representation risks early in the ML pipeline.

---

## Supported Formats

DQDLz supports all major tabular and semi-structured dataset formats:

- **CSV (`.csv`)**: With automatic delimiter sniffing and encoding fallbacks (`utf-8`, `utf-8-sig`, `latin1`, `cp1252`).
- **TSV / TXT (`.tsv`, `.txt`)**: Tab-separated values.
- **Excel (`.xlsx`, `.xls`)**: Single-sheet workbooks loaded via `openpyxl`.
- **JSON (`.json`, `.jsonl`, `.ndjson`)**: Standard JSON arrays, dictionary record wrappers, and newline-delimited JSON flattened automatically via `pd.json_normalize`.
- **Apache Parquet (`.parquet`)**: High-performance columnar data loaded via `pyarrow`.
- **Feather (`.feather`)**: Ultra-fast binary tabular data loaded via `pyarrow`.

---

## Quickstart & Local Usage

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/dqdlz.git
cd dqdlz
pip install -r requirements.txt
```

### Running via Terminal CLI (`main.py`)

Run interactively:
```bash
python main.py
```
Or provide arguments directly:
```bash
python main.py path/to/dataset.parquet target_column
```

### Running via Web API (`api.py`)

Start the FastAPI server locally on port 7860:
```bash
uvicorn api:app --host 0.0.0.0 --port 7860 --reload
```

---

## API Endpoints

### 1. Health Check
```bash
curl http://127.0.0.1:7860/health
```
**Response**:
```json
{
  "status": "ok",
  "version": "0.8.0"
}
```

### 2. Lint Dataset
```bash
curl -X POST "http://127.0.0.1:7860/lint" \
  -F "file=@dataset.csv" \
  -F "target=target"
```

If `API_KEY` is configured in the environment, supply the `X-DQDLZ-KEY` header:
```bash
curl -X POST "http://127.0.0.1:7860/lint" \
  -H "X-DQDLZ-KEY: your_api_key_here" \
  -F "file=@dataset.parquet" \
  -F "target=target"
```

For full documentation of request fields, error codes, and response structures, see [docs/API.md](file:///d:/work/Lint/ml-dataset-lint-main/docs/API.md).

---

## Operational Limits & Configuration

The service is configured through environment variables with safe defaults:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `MAX_UPLOAD_MB` | `25` | Maximum upload file size in megabytes (returns 413 if exceeded). |
| `MAX_COLUMNS` | `2000` | Maximum column count permitted after loading (returns 422 if exceeded). |
| `MAX_CELLS` | `5000000` | Maximum total cell count permitted (`rows * cols`) (returns 422 if exceeded). |
| `MAX_CONCURRENT` | `2` | Maximum concurrent processing requests allowed (returns 429 if busy). |
| `PROCESS_TIMEOUT_SECONDS` | `60` | Maximum processing timeout in seconds (returns 504 if exceeded). |
| `API_KEY` | *(empty)* | Optional secret key; when set, requires `X-DQDLZ-KEY` header on `/lint`. |
| `ALLOWED_ORIGINS` | *(empty)* | Comma-separated list of allowed CORS origins. Empty disables CORS. |
| `ENABLE_DOCS` | `false` | When `true`, enables Swagger `/docs`, ReDoc, and `/openapi.json`. |

---

## Project Structure

```text
ml-dataset-lint/
|-- .github/
|   |-- workflows/
|       |-- ci.yml           # Continuous integration (pytest)
|       |-- deploy-hf.yml    # Automated deploy to Hugging Face Space
|-- core/
|   |-- loader.py            # Multi-format tabular data loader
|   |-- summary.py           # Dataset schema overview & shape
|   |-- checks.py            # Hardened validation & linting rules
|   |-- report.py            # Terminal report formatter
|   |-- linter.py            # Structured linter engine & finding mapper
|-- docs/
|   |-- API.md               # API endpoints, limits, errors, and real payloads
|   |-- CHECKS.md            # Exact rules, categories, codes, and thresholds
|   |-- DATA-HANDLING.md     # Ephemeral storage, privacy, and zero telemetry
|   |-- DEPLOY.md            # Hugging Face deployment, tokens, and rollback guide
|-- space/
|   |-- README.md            # Hugging Face Space deployment metadata
|-- tests/
|   |-- test_api.py          # Pytest suite with FastAPI TestClient
|-- api.py                   # Hardened FastAPI web service
|-- main.py                  # Terminal CLI entry point
|-- Dockerfile               # Hugging Face compliant Docker container
|-- .dockerignore            # Build exclusion rules
|-- requirements.txt         # Pinned production dependencies
|-- README.md                # GitHub repository documentation
|-- LICENSE                  # GNU General Public License v3.0
```

---

## Documentation

- [API Specification](file:///d:/work/Lint/ml-dataset-lint-main/docs/API.md): Endpoints, headers, error tables, and example requests.
- [Lint Rules & Thresholds](file:///d:/work/Lint/ml-dataset-lint-main/docs/CHECKS.md): Catalog of all checks, codes, categories, and exact mathematical thresholds.
- [Data Handling & Privacy](file:///d:/work/Lint/ml-dataset-lint-main/docs/DATA-HANDLING.md): Verified storage, lifecycle, cleanup, and privacy facts.
- [Deployment Guide](file:///d:/work/Lint/ml-dataset-lint-main/docs/DEPLOY.md): Step-by-step setup for Hugging Face Docker Spaces, GitHub Actions, and rollback procedures.

---

## License

This program is free software: you can redistribute it and/or modify it under the terms of the **GNU General Public License (Version 3)** as published by the Free Software Foundation. See the [LICENSE](file:///d:/work/Lint/ml-dataset-lint-main/LICENSE) file for complete details.
