import os
import shutil
import secrets
import asyncio
import logging
import tempfile
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, Query, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError

from core.loader import load_dataset, SUPPORTED_EXTENSIONS
from core.linter import lint_dataset, VERSION

# Configure secure minimal logging (never log user data, content, or API keys)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("dqdlz.api")

# Configuration from Environment Variables with safe defaults
ALLOWED_ORIGINS_ENV = os.getenv("ALLOWED_ORIGINS", "").strip()
API_KEY = os.getenv("API_KEY", "").strip()
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "25"))
MAX_CELLS = int(os.getenv("MAX_CELLS", "5000000"))
MAX_COLUMNS = int(os.getenv("MAX_COLUMNS", "2000"))
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "2"))
PROCESS_TIMEOUT_SECONDS = float(os.getenv("PROCESS_TIMEOUT_SECONDS", "60"))
ENABLE_DOCS = os.getenv("ENABLE_DOCS", "false").lower() in ("true", "1", "yes")

app = FastAPI(
    title="DQDLz API",
    description="DataQuality & DataLintz backend service for tabular ML datasets",
    version=VERSION,
    docs_url="/docs" if ENABLE_DOCS else None,
    redoc_url="/redoc" if ENABLE_DOCS else None,
    openapi_url="/openapi.json" if ENABLE_DOCS else None,
)

# Concurrency semaphore
semaphore = asyncio.Semaphore(MAX_CONCURRENT)

# CORS configuration
if ALLOWED_ORIGINS_ENV:
    origins = [orig.strip() for orig in ALLOWED_ORIGINS_ENV.split(",") if orig.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )


@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    headers = getattr(exc, "headers", None) or {}
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Request validation error: invalid parameters."},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled internal error: %s: %s", type(exc).__name__, str(exc))
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """
    Health check endpoint returning service status and version without touching any files.
    """
    return {
        "status": "ok",
        "version": VERSION,
    }


def _process_dataset_sync(
    file_path: str,
    target: Optional[str],
    thresholds: Dict[str, Any],
    limits_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Synchronous worker executed in executor thread:
    Loads dataset, enforces MAX_COLUMNS and MAX_CELLS, and runs linting.
    """
    df, detected_format = load_dataset(file_path, return_format=True)

    cols = int(df.shape[1])
    if cols > limits_dict["max_columns"]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Dataset column count ({cols}) exceeds maximum allowed limit of {limits_dict['max_columns']}.",
        )

    cells = int(df.shape[0] * cols)
    if cells > limits_dict["max_cells"]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Dataset cell count ({cells}) exceeds maximum allowed limit of {limits_dict['max_cells']}.",
        )

    results = lint_dataset(
        df_or_path=df,
        target=target,
        missing_threshold=thresholds["missing_threshold"],
        dup_threshold=thresholds["dup_threshold"],
        dominance_threshold=thresholds["dominance_threshold"],
        rare_count_threshold=thresholds["rare_count_threshold"],
        corr_threshold=thresholds["corr_threshold"],
        id_ratio=thresholds["id_ratio"],
        min_group_pct=thresholds["min_group_pct"],
        capture_stdout=True,
        limits=limits_dict,
    )
    # Ensure detected format from file is reflected in summary
    results["summary"]["file_format"] = detected_format
    return results


@app.post("/lint", status_code=status.HTTP_200_OK)
async def lint_dataset_endpoint(
    request: Request,
    file: UploadFile = File(..., description="Dataset file (.csv, .tsv, .xlsx, .json, .parquet, .feather)"),
    target: Optional[str] = Form(None, description="Target column name for classification/leakage checks"),
    target_query: Optional[str] = Query(None, alias="target", description="Target column passed via query param"),
    missing_threshold: Optional[float] = Form(30.0, description="Missing value percentage threshold"),
    dup_threshold: Optional[float] = Form(5.0, description="Duplicate rows percentage threshold"),
    dominance_threshold: Optional[float] = Form(90.0, description="Class dominance percentage threshold"),
    rare_count_threshold: Optional[int] = Form(20, description="Rare class sample count threshold"),
    corr_threshold: Optional[float] = Form(0.9, description="Feature-target correlation leakage threshold"),
    id_ratio: Optional[float] = Form(0.95, description="Unique value ratio threshold for ID-like columns"),
    min_group_pct: Optional[float] = Form(10.0, description="Minimum representation percentage for sensitive groups"),
    x_dqdlz_key: Optional[str] = Header(None, alias="X-DQDLZ-KEY"),
):
    """
    Accepts a dataset file upload in any supported tabular format,
    streams and validates payload size, enforces concurrency and execution timeouts,
    and returns comprehensive structured linting findings and statistics.
    """
    # 1. API Key Authentication (constant-time comparison if API_KEY is set)
    if API_KEY:
        if not x_dqdlz_key or not secrets.compare_digest(x_dqdlz_key, API_KEY):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid or missing API key"},
            )

    # 2. Check file extension BEFORE reading the upload body
    raw_filename = file.filename or ""
    _, ext = os.path.splitext(raw_filename)
    ext_lower = ext.lower().strip()
    if not ext_lower or ext_lower not in SUPPORTED_EXTENSIONS:
        supported_str = ", ".join(sorted(SUPPORTED_EXTENSIONS.keys()))
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": f"Unsupported file format '{ext}'. Supported extensions are: {supported_str}"},
        )

    # 3. Concurrency check (non-blocking acquire attempt)
    if semaphore.locked() or semaphore._value <= 0:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "Server busy, try again in a moment"},
            headers={"Retry-After": "5"},
        )

    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=0.05)
    except (asyncio.TimeoutError, Exception):
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "Server busy, try again in a moment"},
            headers={"Retry-After": "5"},
        )

    # Prepare per-request temporary directory
    temp_dir = tempfile.mkdtemp(prefix="dqdlz_")
    temp_file_path = os.path.join(temp_dir, f"dataset{ext_lower}")

    try:
        # 4. Stream upload with size enforcement (abort early on > MAX_UPLOAD_MB)
        max_bytes = int(MAX_UPLOAD_MB * 1024 * 1024)
        bytes_read = 0

        with open(temp_file_path, "wb") as buffer:
            while True:
                chunk = await file.read(65536)
                if not chunk:
                    break
                bytes_read += len(chunk)
                if bytes_read > max_bytes:
                    return JSONResponse(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        content={"detail": f"File exceeds maximum allowed upload size of {MAX_UPLOAD_MB} MB."},
                    )
                buffer.write(chunk)

        effective_target = target if target is not None else target_query
        if effective_target is not None:
            effective_target = effective_target.strip()
            if not effective_target:
                effective_target = None

        thresholds = {
            "missing_threshold": missing_threshold if missing_threshold is not None else 30.0,
            "dup_threshold": dup_threshold if dup_threshold is not None else 5.0,
            "dominance_threshold": dominance_threshold if dominance_threshold is not None else 90.0,
            "rare_count_threshold": rare_count_threshold if rare_count_threshold is not None else 20,
            "corr_threshold": corr_threshold if corr_threshold is not None else 0.9,
            "id_ratio": id_ratio if id_ratio is not None else 0.95,
            "min_group_pct": min_group_pct if min_group_pct is not None else 10.0,
        }

        limits_dict = {
            "max_upload_mb": MAX_UPLOAD_MB,
            "max_cells": MAX_CELLS,
            "max_columns": MAX_COLUMNS,
            "process_timeout_seconds": PROCESS_TIMEOUT_SECONDS,
        }

        # 5. Execute processing in thread pool with timeout enforcement
        loop = asyncio.get_running_loop()
        try:
            results = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    _process_dataset_sync,
                    temp_file_path,
                    effective_target,
                    thresholds,
                    limits_dict,
                ),
                timeout=PROCESS_TIMEOUT_SECONDS,
            )
            return JSONResponse(status_code=status.HTTP_200_OK, content=results)

        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                content={"detail": f"Dataset processing timed out after {PROCESS_TIMEOUT_SECONDS} seconds."},
            )
        except HTTPException as he:
            return JSONResponse(status_code=he.status_code, content={"detail": he.detail})
        except Exception as exc:
            logger.warning("Dataset parsing or linting failure: %s", type(exc).__name__)
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": f"Failed to process dataset: {str(exc)}"},
            )

    finally:
        # Guarantee cleanup of temporary directory and release of semaphore
        shutil.rmtree(temp_dir, ignore_errors=True)
        semaphore.release()
