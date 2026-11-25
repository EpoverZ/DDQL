# DQDLz Data Handling & Privacy Policy

This document outlines the precise lifecycle of user data uploaded to the DQDLz service, verified directly from the code implementation.

---

## 1. Upload Lifecycle

1. **Pre-flight Extension Verification**:
   - The uploaded file's extension is inspected *before* reading the request body. If the extension is not supported, the request is rejected immediately with HTTP 400 and no disk writes occur.
2. **Ephemeral Disk Streaming**:
   - The file payload is streamed in 64 KB chunks directly into a dedicated, per-request temporary directory created via `tempfile.mkdtemp(prefix="dqdlz_")`.
   - Upload size is actively tracked during streaming. If total bytes exceed `MAX_UPLOAD_MB` (default: 25 MB), streaming is aborted immediately with HTTP 413, and the partial file is purged.
3. **In-Memory Analysis**:
   - The file is parsed into a temporary in-memory `pandas.DataFrame`.
   - Size limits (`MAX_COLUMNS`, `MAX_CELLS`) are enforced before linting checks execute.
4. **Guaranteed Purge in `finally` Block**:
   - Regardless of whether the linting succeeds, fails, encounters parsing errors, or times out, the entire temporary directory is completely deleted in the endpoint's `finally:` block using `shutil.rmtree(temp_dir, ignore_errors=True)`.
   - No uploaded dataset files persist on disk after the HTTP request terminates.

---

## 2. Telemetry, Logging & Data Sanitization

- **No User Data Logging**:
  - The application logs never record file contents, row records, column names, cell values, or the `X-DQDLZ-KEY` authentication header.
  - Server logs only record the file extension (e.g. `.parquet`, `.csv`), request duration, and HTTP status codes.
- **Console Capture**:
  - Standard output prints emitted during lint checks are captured in an in-memory `io.StringIO` buffer and discarded, preventing user dataset values or column names from leaking to stdout or container logs.

---

## 3. Third-Party Integrations

- **Zero External Telemetry**:
  - DQDLz does not connect to external analytics, tracking, or cloud storage providers.
  - All dataset parsing, summary generation, statistical validations, and leakage checks are executed 100% locally within the isolated container process.
