import os
import sys
import json
import tempfile
import pytest
import pandas as pd
import numpy as np

from main import run_cli, build_parser
from core.linter import lint_dataset


@pytest.fixture
def sample_dataframe():
    """
    Creates a representative dataset with various data types, missing values,
    duplicates, sensitive attributes, and target.
    """
    n = 50
    return pd.DataFrame({
        "id": list(range(n)),
        "feature_num": [float(i * 2) for i in range(n)],
        "gender": ["male" if i % 10 != 0 else "other" for i in range(n)],
        "missing_col": [None if i < 20 else float(i) for i in range(n)],  # 40% missing
        "target": [0 if i < 45 else 1 for i in range(n)],  # 90% dominance
    })


def test_argparse_help():
    """Verify that --help parses cleanly and provides usage information."""
    parser = build_parser()
    help_text = parser.format_help()
    assert "dqdlz" in help_text
    assert "--target" in help_text
    assert "--json" in help_text
    assert "--missing-threshold" in help_text
    assert "--dup-threshold" in help_text
    assert "--corr-threshold" in help_text


def test_cli_csv_terminal_output(sample_dataframe, capsys):
    """Test running CLI on CSV file in human-readable terminal mode."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="", encoding="utf-8") as f:
        sample_dataframe.to_csv(f.name, index=False)
        temp_path = f.name

    try:
        exit_code = run_cli([temp_path])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "Dataset Summary" in captured.out
        assert "Format: csv" in captured.out
        assert "No target provided -- skipping target-based checks" in captured.out
        assert "LINT WARNINGS" in captured.out
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_cli_with_target(sample_dataframe, capsys):
    """Test CLI flag mode with --target specified."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="", encoding="utf-8") as f:
        sample_dataframe.to_csv(f.name, index=False)
        temp_path = f.name

    try:
        exit_code = run_cli([temp_path, "--target", "target"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "Target preview:" in captured.out
        assert "Class counts:" in captured.out
        assert "LINT WARNINGS" in captured.out
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_cli_json_output(sample_dataframe, capsys):
    """Test --json flag produces valid structured JSON output."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="", encoding="utf-8") as f:
        sample_dataframe.to_csv(f.name, index=False)
        temp_path = f.name

    try:
        exit_code = run_cli([temp_path, "--target", "target", "--json"])
        captured = capsys.readouterr()

        assert exit_code == 0
        data = json.loads(captured.out)
        assert data["status"] == "ok"
        assert "summary" in data
        assert "findings" in data
        assert "stats" in data
        assert data["summary"]["file_format"] == "csv"
        assert data["stats"]["target_analysis"]["column"] == "target"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_supported_formats_roundtrip(sample_dataframe, capsys):
    """Test CLI against multiple tabular formats (.csv, .xlsx, .json, .parquet)."""
    formats = {
        ".csv": lambda df, path: df.to_csv(path, index=False),
        ".xlsx": lambda df, path: df.to_excel(path, index=False),
        ".json": lambda df, path: df.to_json(path, orient="records"),
        ".parquet": lambda df, path: df.to_parquet(path, index=False),
    }

    for ext, save_fn in formats.items():
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            temp_path = f.name

        try:
            save_fn(sample_dataframe, temp_path)
            exit_code = run_cli([temp_path, "--json"])
            captured = capsys.readouterr()

            assert exit_code == 0, f"Failed for format {ext}"
            data = json.loads(captured.out)
            assert data["status"] == "ok"
            assert data["summary"]["rows"] == len(sample_dataframe)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


def test_custom_thresholds(capsys):
    """Test configuring thresholds via CLI flags."""
    df = pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 4.0, 5.0, None],  # ~16.67% missing
        "target": [0, 0, 0, 0, 1, 1],
    })

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="", encoding="utf-8") as f:
        df.to_csv(f.name, index=False)
        temp_path = f.name

    try:
        # With default missing-threshold=30.0, 16.7% missing should NOT warn
        run_cli([temp_path, "--missing-threshold", "30.0", "--json"])
        data_default = json.loads(capsys.readouterr().out)
        missing_findings_default = [
            f for f in data_default["findings"] if f["code"] == "missing_values_threshold_exceeded"
        ]
        assert len(missing_findings_default) == 0

        # With lowered missing-threshold=10.0, 16.7% missing MUST warn
        run_cli([temp_path, "--missing-threshold", "10.0", "--json"])
        data_lowered = json.loads(capsys.readouterr().out)
        missing_findings_lowered = [
            f for f in data_lowered["findings"] if f["code"] == "missing_values_threshold_exceeded"
        ]
        assert len(missing_findings_lowered) == 1
        assert missing_findings_lowered[0]["columns"] == ["num"]
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_interactive_mode_fallback(sample_dataframe, monkeypatch, capsys):
    """Test interactive input prompts when no positional file argument is passed."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="", encoding="utf-8") as f:
        sample_dataframe.to_csv(f.name, index=False)
        temp_path = f.name

    try:
        inputs = iter([temp_path, "target"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

        exit_code = run_cli([])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "Dataset Summary" in captured.out
        assert "Target preview:" in captured.out
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_interactive_empty_input(monkeypatch, capsys):
    """Test interactive mode when user submits empty file path."""
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    exit_code = run_cli([])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "[ERROR] No dataset path provided." in captured.out


def test_invalid_file_path(capsys):
    """Test providing non-existent file path returns error code 1."""
    exit_code = run_cli(["non_existent_dataset.csv"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "[ERROR]" in captured.out


def test_unsupported_file_extension(capsys):
    """Test providing an unsupported extension returns error code 1."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        temp_path = f.name

    try:
        exit_code = run_cli([temp_path])
        captured = capsys.readouterr()
        assert exit_code == 1
        assert "Unsupported file format" in captured.out
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_invalid_target_column(sample_dataframe, capsys):
    """Test specifying a target column that does not exist in the dataset."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="", encoding="utf-8") as f:
        sample_dataframe.to_csv(f.name, index=False)
        temp_path = f.name

    try:
        exit_code = run_cli([temp_path, "--target", "non_existent_target"])
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "Target column 'non_existent_target' not found" in captured.out
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
