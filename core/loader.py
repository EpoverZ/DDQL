# =============================#
#        Load Dataset          #
# =============================#

import os
import json
from typing import Tuple, Optional
import pandas as pd

SUPPORTED_EXTENSIONS = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".txt": "tsv",
    ".xlsx": "excel",
    ".xls": "excel",
    ".json": "json",
    ".jsonl": "jsonl",
    ".ndjson": "jsonl",
    ".parquet": "parquet",
    ".feather": "feather",
}


def detect_file_format(path: str) -> str:
    """
    Detects the normalized dataset format string based on the file extension.
    Raises ValueError for unsupported formats.
    """
    _, ext = os.path.splitext(path)
    ext_lower = ext.lower().strip()
    if not ext_lower or ext_lower not in SUPPORTED_EXTENSIONS:
        supported_str = ", ".join(sorted(SUPPORTED_EXTENSIONS.keys()))
        raise ValueError(
            f"Unsupported file format '{ext}'. Supported extensions are: {supported_str}"
        )
    return SUPPORTED_EXTENSIONS[ext_lower]


def _read_csv(path: str) -> pd.DataFrame:
    """
    Loads CSV with automatic delimiter sniffing and encoding fallbacks.
    """
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    for enc in encodings:
        try:
            # First try default comma delimiter
            df = pd.read_csv(path, encoding=enc)
            # If loaded as a single column with possible semicolons, try delimiter sniffing
            if df.shape[1] == 1:
                try:
                    df_sniff = pd.read_csv(path, sep=None, engine="python", encoding=enc)
                    if df_sniff.shape[1] > 1:
                        return df_sniff
                except Exception:
                    pass
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to decode CSV file '{path}' using supported encodings.")


def _read_tsv(path: str) -> pd.DataFrame:
    """
    Loads TSV/TXT tab-separated dataset with encoding fallbacks.
    """
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    for enc in encodings:
        try:
            return pd.read_csv(path, sep="\t", encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to decode TSV/TXT file '{path}' using supported encodings.")


def _read_json(path: str, is_lines: bool = False) -> pd.DataFrame:
    """
    Loads JSON / JSONL / NDJSON files and flattens nested objects with pd.json_normalize.
    """
    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    content = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                content = f.read().strip()
            break
        except UnicodeDecodeError:
            continue

    if content is None:
        raise ValueError(f"Unable to read JSON file '{path}' with supported encodings.")

    if not content:
        raise ValueError(f"JSON file '{path}' is empty.")

    if is_lines:
        records = []
        for line in content.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return pd.json_normalize(records)

    # Standard JSON handling
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Fallback: maybe file has .json extension but newline-delimited lines
        try:
            records = [json.loads(line) for line in content.splitlines() if line.strip()]
            if records:
                return pd.json_normalize(records)
        except Exception:
            pass
        raise

    if isinstance(data, list):
        return pd.json_normalize(data)
    elif isinstance(data, dict):
        # Look for wrapped lists: e.g. {"data": [...]}, {"records": [...]}, {"items": [...]}
        for key in ["data", "records", "items", "rows", "results"]:
            if key in data and isinstance(data[key], list) and len(data[key]) > 0:
                if isinstance(data[key][0], dict):
                    return pd.json_normalize(data[key])
        for val in data.values():
            if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                return pd.json_normalize(val)
        # Fallback to normalizing the dict directly
        try:
            df = pd.json_normalize(data)
            if not df.empty:
                return df
        except Exception:
            pass
        return pd.read_json(path)
    else:
        raise ValueError(f"JSON content in '{path}' must be an object or array of objects.")


def load_dataset(path: str, return_format: bool = False):
    """
    Loads a dataset from any supported tabular format:
    .csv, .tsv, .txt, .xlsx, .xls, .json, .jsonl, .ndjson, .parquet, .feather.
    Returns pd.DataFrame (or (pd.DataFrame, format_str) if return_format is True).
    """
    fmt = detect_file_format(path)

    if fmt == "csv":
        df = _read_csv(path)
    elif fmt == "tsv":
        df = _read_tsv(path)
    elif fmt == "excel":
        df = pd.read_excel(path)
    elif fmt == "json":
        df = _read_json(path, is_lines=False)
    elif fmt == "jsonl":
        df = _read_json(path, is_lines=True)
    elif fmt == "parquet":
        df = pd.read_parquet(path)
    elif fmt == "feather":
        df = pd.read_feather(path)
    else:
        raise ValueError(f"Unhandled file format: {fmt}")

    if return_format:
        return df, fmt
    return df
