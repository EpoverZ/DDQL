# DQDLz Deployment Guide (Hugging Face Docker Space)

This guide documents the complete deployment architecture for running DQDLz as a Docker Space on Hugging Face, utilizing the GitHub repository as the single source of truth.

---

## 1. Prerequisites & Hugging Face Setup

### Step 1: Create the Space on Hugging Face
1. Log in to your [Hugging Face](https://huggingface.co) account.
2. Click **New Space** (`https://huggingface.co/new-space`).
3. Set the Space name (e.g. `dqdlz-api`).
4. Select the License: **GNU General Public License v3.0**.
5. Choose **Docker** as the Space SDK (Blank template).
6. Select the hardware tier (Default **CPU basic · 2 vCPU · 16 GB RAM · Free**).
7. Create the Space.

### Step 2: Generate Hugging Face Access Token
1. Go to **Settings -> Access Tokens** (`https://huggingface.co/settings/tokens`).
2. Click **Create new token**.
3. Name: `github-actions-space-deploy`.
4. Role: **Write**.
5. Copy the generated token string.

### Step 3: Configure GitHub Repository Secrets & Variables
In your GitHub repository, navigate to **Settings -> Secrets and variables -> Actions**:

1. **Repository Secret**:
   - Name: `HF_TOKEN`
   - Value: The Hugging Face Write Token copied in Step 2.
2. **Repository Variable**:
   - Name: `HF_SPACE_ID`
   - Value: `<hf-username>/<space-name>` (e.g., `acme/dqdlz-api`).

---

## 2. Deployment Workflow (`.github/workflows/deploy-hf.yml`)

The automated deployment pipeline triggers on every push to the `main` branch or via manual `workflow_dispatch`:

1. **Continuous Integration**: Executes `pytest` against all supported dataset formats and edge cases.
2. **Space Artifact Assembly**: Creates an isolated `build/space/` staging folder:
   - Copies `api.py`, `main.py`, `core/`, `Dockerfile`, `requirements.txt`, `.dockerignore`, and `LICENSE`.
   - Copies `space/README.md` to `build/space/README.md` to supply the Hugging Face Space metadata front matter.
3. **Payload Verification**: Scans `build/space/` and fails loudly if any file exceeds 10 MB.
4. **Synchronous Upload**: Invokes `huggingface_hub.HfApi().upload_folder(..., delete_patterns=['*'])` to ensure removed files in the repository are cleaned from the Space.
5. **Health Verification**: Polls the Space until it reaches `RUNNING` stage and validates `GET /health`.

---

## 3. Rollback Procedure

In the event of a breaking deployment:
1. **GitHub Revert (Recommended)**:
   ```bash
   git revert HEAD
   git push origin main
   ```
   This automatically triggers the test suite and redeploys the last known good commit.
2. **Manual Dispatch**: Trigger the `Deploy to Hugging Face Space` action manually from a tagged release or stable commit branch.
3. **Hugging Face Commit Reset**: In the Hugging Face Space repository tab, you can reset the git tree to any earlier commit hash.

---

## 4. Threat Model & Untrusted Upload Handling

DQDLz treats all uploaded files as untrusted:

- **No Deserialization Attacks**:
  - `pickle`, `cPickle`, and `joblib` deserializers are completely prohibited.
  - No `eval()` or `exec()` is executed on user input.
- **Formula Injection Mitigation**:
  - Excel files are read strictly as tabular data values via `openpyxl`.
- **DoS & Resource Exhaustion Defenses**:
  - `MAX_UPLOAD_MB` (default 25 MB): Streamed upload terminates early on oversized files.
  - `MAX_COLUMNS` (default 2,000) & `MAX_CELLS` (default 5,000,000): Rejects bloated datasets or JSON objects that expand excessively under flattening.
  - `MAX_CONCURRENT` (default 2): Enforces worker throttling, rejecting excess requests with HTTP 429 and `Retry-After: 5`.
  - `PROCESS_TIMEOUT_SECONDS` (default 60s): Cancels the async request with HTTP 504 if computing checks takes longer than the timeout.

### Remaining Risks & Thread Timeout Limitations
- **CPU-bound Pandas Operations**: In CPython, CPU-intensive C-extensions (such as dense Pearson correlation matrices or large factorizations) execute in GIL-bound threads that cannot be preemptively terminated from asyncio. While `asyncio.wait_for` returns `504 Gateway Timeout` to the HTTP client immediately, the thread will finish its current CPU cycle. The `MAX_CONCURRENT` semaphore ensures that even under repeated timeouts, unconstrained threads cannot multiply and saturate the server.
