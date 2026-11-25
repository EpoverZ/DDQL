import io
import os
import json
import tempfile
import asyncio
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

import api
from api import app


@pytest.fixture
def client():
    # Return TestClient
    return TestClient(app)


def test_health_endpoint(client):
    """
    GET /health returns 200 with status ok and version, touching no files.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert data["version"] == api.VERSION


def test_supported_formats(client):
    """
    Test every supported format: .csv, .tsv, .xlsx, .json, .parquet, .feather.
    All must return 200 OK.
    """
    n = 20
    df = pd.DataFrame({
        "id": list(range(n)),
        "feature_num": [float(i * 2) for i in range(n)],
        "target": [0 if i < 15 else 1 for i in range(n)]
    })

    # 1. CSV
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    resp = client.post("/lint", files={"file": ("data.csv", csv_bytes, "text/csv")})
    assert resp.status_code == 200
    assert resp.json()["summary"]["file_format"] == "csv"

    # 2. TSV
    tsv_bytes = df.to_csv(index=False, sep="\t").encode("utf-8")
    resp = client.post("/lint", files={"file": ("data.tsv", tsv_bytes, "text/tab-separated-values")})
    assert resp.status_code == 200
    assert resp.json()["summary"]["file_format"] == "tsv"

    # 3. Excel (.xlsx)
    xlsx_buf = io.BytesIO()
    df.to_excel(xlsx_buf, index=False)
    resp = client.post("/lint", files={"file": ("data.xlsx", xlsx_buf.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
    assert resp.status_code == 200
    assert resp.json()["summary"]["file_format"] == "excel"

    # 4. JSON
    json_bytes = df.to_json(orient="records").encode("utf-8")
    resp = client.post("/lint", files={"file": ("data.json", json_bytes, "application/json")})
    assert resp.status_code == 200
    assert resp.json()["summary"]["file_format"] == "json"

    # 5. Parquet
    parquet_buf = io.BytesIO()
    df.to_parquet(parquet_buf, index=False)
    resp = client.post("/lint", files={"file": ("data.parquet", parquet_buf.getvalue(), "application/octet-stream")})
    assert resp.status_code == 200
    assert resp.json()["summary"]["file_format"] == "parquet"

    # 6. Feather
    feather_buf = io.BytesIO()
    df.to_feather(feather_buf)
    resp = client.post("/lint", files={"file": ("data.feather", feather_buf.getvalue(), "application/octet-stream")})
    assert resp.status_code == 200
    assert resp.json()["summary"]["file_format"] == "feather"


def test_response_fields_backward_compatibility_and_additive(client):
    """
    Verify all old keys are preserved unchanged, and new additive fields are present.
    """
    df = pd.DataFrame({"a": [1, 2, 2], "b": ["x", "y", "y"]})
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    resp = client.post("/lint", files={"file": ("test.csv", csv_bytes, "text/csv")})
    assert resp.status_code == 200
    data = resp.json()

    # Pre-existing keys MUST be preserved
    assert "status" in data
    assert "summary" in data
    assert "warnings" in data
    assert "errors" in data
    assert "stats" in data
    assert isinstance(data["summary"], dict)
    assert isinstance(data["warnings"], list)
    assert isinstance(data["errors"], list)
    assert isinstance(data["stats"], dict)

    # Summary keys
    assert "rows" in data["summary"]
    assert "columns" in data["summary"]
    assert "column_types" in data["summary"]
    assert "file_format" in data["summary"]

    # Stats keys
    assert "missing_values" in data["stats"]
    assert "duplicate_rows" in data["stats"]
    assert "constant_columns" in data["stats"]
    assert "target_analysis" in data["stats"]

    # New additive keys
    assert "version" in data
    assert "findings" in data
    assert "limits" in data
    assert isinstance(data["findings"], list)
    assert isinstance(data["limits"], dict)

    # Check structure of each finding object
    for finding in data["findings"]:
        assert "level" in finding
        assert finding["level"] in ("warn", "info")
        assert "category" in finding
        assert finding["category"] in ("data_quality", "target", "leakage", "bias", "trainability")
        assert "code" in finding
        assert "message" in finding
        assert "columns" in finding
        assert isinstance(finding["columns"], list)


def test_unsupported_format(client):
    """
    Uploading a file with unsupported extension must return 400.
    """
    resp = client.post(
        "/lint",
        files={"file": ("sample.pdf", b"%PDF-1.4 dummy", "application/pdf")},
    )
    assert resp.status_code == 400
    assert "Unsupported file format" in resp.json()["detail"]


def test_corrupted_file(client):
    """
    Uploading corrupted content must return 400.
    """
    resp = client.post(
        "/lint",
        files={"file": ("corrupted.json", b"{invalid json content!", "application/json")},
    )
    assert resp.status_code == 400
    assert "Failed to process dataset" in resp.json()["detail"]


def test_oversize_file_413(client, monkeypatch):
    """
    Uploading a file exceeding MAX_UPLOAD_MB must return 413.
    """
    # Temporarily set MAX_UPLOAD_MB to 0.001 MB (~1KB)
    monkeypatch.setattr(api, "MAX_UPLOAD_MB", 0.001)

    payload = b"col1,col2\n" + (b"val1,val2\n" * 200) # > 1KB
    resp = client.post(
        "/lint",
        files={"file": ("large.csv", payload, "text/csv")},
    )
    assert resp.status_code == 413
    assert "maximum allowed upload size" in resp.json()["detail"]


def test_max_columns_exceeded_422(client, monkeypatch):
    """
    Dataset exceeding MAX_COLUMNS must return 422.
    """
    monkeypatch.setattr(api, "MAX_COLUMNS", 5)

    # Create dataset with 10 columns
    df = pd.DataFrame({f"col_{i}": [1, 2] for i in range(10)})
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    resp = client.post(
        "/lint",
        files={"file": ("many_cols.csv", csv_bytes, "text/csv")},
    )
    assert resp.status_code == 422
    assert "Dataset column count" in resp.json()["detail"]


def test_max_cells_exceeded_422(client, monkeypatch):
    """
    Dataset exceeding MAX_CELLS must return 422.
    """
    monkeypatch.setattr(api, "MAX_CELLS", 50)

    # Create dataset with 100 cells (10 rows x 10 cols)
    df = pd.DataFrame({f"col_{i}": list(range(10)) for i in range(10)})
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    resp = client.post(
        "/lint",
        files={"file": ("many_cells.csv", csv_bytes, "text/csv")},
    )
    assert resp.status_code == 422
    assert "Dataset cell count" in resp.json()["detail"]


def test_busy_case_429(client, monkeypatch):
    """
    When the concurrency semaphore is busy, the endpoint returns 429 with Retry-After header.
    """
    # Create an already locked semaphore
    busy_semaphore = asyncio.Semaphore(0)
    monkeypatch.setattr(api, "semaphore", busy_semaphore)

    csv_bytes = b"a,b\n1,2\n"
    resp = client.post(
        "/lint",
        files={"file": ("test.csv", csv_bytes, "text/csv")},
    )
    assert resp.status_code == 429
    assert resp.json()["detail"] == "Server busy, try again in a moment"
    assert "retry-after" in [h.lower() for h in resp.headers.keys()]


def test_timeout_case_504(client, monkeypatch):
    """
    When processing times out, return 504.
    """
    monkeypatch.setattr(api, "PROCESS_TIMEOUT_SECONDS", 0.001)

    def slow_process(*args, **kwargs):
        import time
        time.sleep(0.05)
        return {}

    monkeypatch.setattr(api, "_process_dataset_sync", slow_process)

    csv_bytes = b"a,b\n1,2\n"
    resp = client.post(
        "/lint",
        files={"file": ("test.csv", csv_bytes, "text/csv")},
    )
    assert resp.status_code == 504
    assert "timed out" in resp.json()["detail"]


def test_api_key_required_when_set(client, monkeypatch):
    """
    When API_KEY is set:
    - Without header or wrong key -> 401
    - With correct header -> 200
    - /health stays open without key
    """
    monkeypatch.setattr(api, "API_KEY", "secret123")

    # /health stays open
    health_resp = client.get("/health")
    assert health_resp.status_code == 200

    csv_bytes = b"a,b\n1,2\n"

    # Missing API key
    resp_no_key = client.post(
        "/lint",
        files={"file": ("test.csv", csv_bytes, "text/csv")},
    )
    assert resp_no_key.status_code == 401
    assert resp_no_key.json()["detail"] == "Invalid or missing API key"

    # Wrong API key
    resp_wrong_key = client.post(
        "/lint",
        files={"file": ("test.csv", csv_bytes, "text/csv")},
        headers={"X-DQDLZ-KEY": "wrong_secret"},
    )
    assert resp_wrong_key.status_code == 401

    # Correct API key
    resp_correct = client.post(
        "/lint",
        files={"file": ("test.csv", csv_bytes, "text/csv")},
        headers={"X-DQDLZ-KEY": "secret123"},
    )
    assert resp_correct.status_code == 200


def test_cors_origins():
    """
    Verify CORS works only for allowed origins.
    """
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    # Test with restricted origin
    test_app = FastAPI()
    test_app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://allowed.example.com"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @test_app.get("/health")
    def h():
        return {"status": "ok"}

    c = TestClient(test_app)

    # Disallowed origin
    resp_disallowed = c.get("/health", headers={"Origin": "https://attacker.example.com"})
    assert resp_disallowed.headers.get("access-control-allow-origin") is None

    # Allowed origin
    resp_allowed = c.get("/health", headers={"Origin": "https://allowed.example.com"})
    assert resp_allowed.headers.get("access-control-allow-origin") == "https://allowed.example.com"


def test_temp_files_removed_afterwards(client):
    """
    Verify that temporary files and directories created during processing are removed afterwards.
    """
    temp_root = tempfile.gettempdir()
    before_dirs = set(os.listdir(temp_root))

    csv_bytes = b"a,b\n1,2\n3,4\n"
    resp = client.post("/lint", files={"file": ("test.csv", csv_bytes, "text/csv")})
    assert resp.status_code == 200

    after_dirs = set(os.listdir(temp_root))
    new_dirs = [d for d in after_dirs - before_dirs if d.startswith("dqdlz_")]
    assert len(new_dirs) == 0, f"Temporary directories were leaked: {new_dirs}"


def test_unhashable_columns(client):
    """
    Verify datasets containing unhashable objects (e.g. nested lists and dicts in JSON)
    execute checks without crashing.
    """
    records = [
        {"id": 1, "tags": ["red", "blue"], "meta": {"owner": "admin"}, "target": 0},
        {"id": 2, "tags": ["green"], "meta": {"owner": "user"}, "target": 1},
        {"id": 1, "tags": ["red", "blue"], "meta": {"owner": "admin"}, "target": 0}, # duplicate
    ]
    json_bytes = json.dumps(records).encode("utf-8")

    resp = client.post("/lint", files={"file": ("unhashable.json", json_bytes, "application/json")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["stats"]["duplicate_rows"]["count"] >= 1
    assert len(data["findings"]) > 0
