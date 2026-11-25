# DQDLz API Documentation

DQDLz (DataQuality & DataLintz) provides a lightweight, hardened, high-throughput REST API for automated structural, quality, target, and leakage validation of tabular machine learning datasets.

---

## Base Endpoints

| Method | Endpoint | Description | Auth Required |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Service health and active version check. Touches no files. | No |
| `POST` | `/lint` | Upload dataset file and execute multi-phase linting analysis. | Optional (`API_KEY`) |

---

## Authentication & Headers

- **`X-DQDLZ-KEY`**: When the service environment has `API_KEY` configured, every request to `POST /lint` must supply this header matching the secret key (verified via constant-time comparison). `GET /health` remains publicly accessible without a key.
- **`Retry-After`**: Returned in HTTP 429 responses indicating the number of seconds a client should wait before retrying (default: 5 seconds).
- **`Origin`**: Handled via standard CORS middleware based on the `ALLOWED_ORIGINS` environment variable.

---

## Request Fields (`POST /lint`)

The endpoint accepts `multipart/form-data`:

| Parameter | Type | Required | Description | Default |
| :--- | :--- | :---: | :--- | :--- |
| `file` | File (`UploadFile`) | **Yes** | Tabular dataset file (`.csv`, `.tsv`, `.txt`, `.xlsx`, `.xls`, `.json`, `.jsonl`, `.ndjson`, `.parquet`, `.feather`). | None |
| `target` | string (Form / Query) | No | Target column name for classification/leakage analysis. | `None` |
| `missing_threshold` | float (Form) | No | Missing value percentage threshold for column warnings. | `30.0` |
| `dup_threshold` | float (Form) | No | Duplicate rows percentage threshold for dataset warning. | `5.0` |
| `dominance_threshold` | float (Form) | No | Label percentage threshold triggering dominance warning. | `90.0` |
| `rare_count_threshold`| int (Form) | No | Minimum sample count below which a target class is rare. | `20` |
| `corr_threshold` | float (Form) | No | Feature-target numeric correlation leakage threshold. | `0.9` |
| `id_ratio` | float (Form) | No | Unique ratio threshold flagging potential ID-like columns. | `0.95` |
| `min_group_pct` | float (Form) | No | Representation threshold flagging under-represented groups. | `10.0` |

---

## Operational Limits & Guardrails

The service enforces strict environment-driven limits:

| Limit Name | Env Variable | Default | Action When Exceeded |
| :--- | :--- | :--- | :--- |
| Max Upload Size | `MAX_UPLOAD_MB` | `25` MB | Aborts streaming early with `HTTP 413 Payload Too Large`. |
| Max Columns | `MAX_COLUMNS` | `2,000` columns | Rejects dataset with `HTTP 422 Unprocessable Entity`. |
| Max Total Cells | `MAX_CELLS` | `5,000,000` cells | Rejects dataset with `HTTP 422 Unprocessable Entity`. |
| Max Concurrency | `MAX_CONCURRENT` | `2` workers | Rejects excess concurrent requests with `HTTP 429 Too Many Requests`. |
| Execution Timeout | `PROCESS_TIMEOUT_SECONDS` | `60` seconds | Cancels thread and returns `HTTP 504 Gateway Timeout`. |

---

## HTTP Error Status Codes

All errors return a clean JSON payload of shape `{"detail": "<message>"}`:

| Status Code | Reason | Example Response |
| :--- | :--- | :--- |
| **`400 Bad Request`** | Unsupported extension or corrupt/unparseable file content. | `{"detail": "Unsupported file format '.pdf'. Supported extensions are: .csv, .feather, .json, .jsonl, .ndjson, .parquet, .tsv, .txt, .xls, .xlsx"}` |
| **`401 Unauthorized`** | Missing or invalid `X-DQDLZ-KEY` when `API_KEY` is configured. | `{"detail": "Invalid or missing API key"}` |
| **`413 Payload Too Large`** | Upload stream exceeded `MAX_UPLOAD_MB`. | `{"detail": "File exceeds maximum allowed upload size of 25.0 MB."}` |
| **`422 Unprocessable Entity`** | Exceeded `MAX_COLUMNS` or `MAX_CELLS` after loading. | `{"detail": "Dataset cell count (6000000) exceeds maximum allowed limit of 5000000."}` |
| **`429 Too Many Requests`** | Server concurrency limit reached (`MAX_CONCURRENT`). | `{"detail": "Server busy, try again in a moment"}` (with `Retry-After: 5`) |
| **`504 Gateway Timeout`** | Processing exceeded `PROCESS_TIMEOUT_SECONDS`. | `{"detail": "Dataset processing timed out after 60.0 seconds."}` |
| **`500 Internal Server Error`** | Uncaught server exception (no internal stack trace exposed). | `{"detail": "Internal server error"}` |

---

## Real Example Request & Response

### Request (`curl`)
```bash
curl -X POST "http://127.0.0.1:7860/lint" \
  -H "X-DQDLZ-KEY: test" \
  -F "file=@sample.csv" \
  -F "target=target"
```

### Response (`HTTP 200 OK`)
```json
{
  "status": "ok",
  "version": "0.8.0",
  "summary": {
    "file_format": "csv",
    "rows": 30,
    "columns": 3,
    "column_types": {
      "id": "int64",
      "feat_num": "float64",
      "target": "int64"
    }
  },
  "warnings": [
    "[WARN] Label '1' has only 5 samples",
    "[WARN] Feature 'feat_num' highly correlated with target (0.95)",
    "[WARN] Feature 'id' appears ID-like",
    "[INFO] Dataset is very small (<100 rows)"
  ],
  "findings": [
    {
      "level": "warn",
      "category": "target",
      "code": "rare_class",
      "message": "[WARN] Label '1' has only 5 samples",
      "columns": [
        "target"
      ]
    },
    {
      "level": "warn",
      "category": "leakage",
      "code": "high_target_correlation",
      "message": "[WARN] Feature 'feat_num' highly correlated with target (0.95)",
      "columns": [
        "feat_num",
        "target"
      ]
    },
    {
      "level": "warn",
      "category": "leakage",
      "code": "id_like_column",
      "message": "[WARN] Feature 'id' appears ID-like",
      "columns": [
        "id"
      ]
    },
    {
      "level": "info",
      "category": "data_quality",
      "code": "small_dataset",
      "message": "[INFO] Dataset is very small (<100 rows)",
      "columns": []
    }
  ],
  "errors": [],
  "stats": {
    "missing_values": {},
    "duplicate_rows": {
      "count": 0,
      "percentage": 0.0
    },
    "constant_columns": [],
    "target_analysis": {
      "column": "target",
      "class_frequencies": {
        "0": 25,
        "1": 5
      },
      "class_percentages": {
        "0": 83.33,
        "1": 16.67
      }
    }
  },
  "limits": {
    "max_upload_mb": 25.0,
    "max_cells": 5000000,
    "max_columns": 2000,
    "process_timeout_seconds": 60.0
  }
}
```
