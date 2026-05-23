# ── Stop on errors ────────────────────────────────────────────────────────────
$ErrorActionPreference = "Stop"

# ── Step 1: Check Python >= 3.11 ─────────────────────────────────────────────
Write-Host "Checking Python version..."

$PYTHON_CMD = "python"

$version = & $PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"

$major = [int]($version.Split('.')[0])
$minor = [int]($version.Split('.')[1])

if ($major -lt 3 -or $minor -lt 11) {
    Write-Host "ERROR: Python 3.11 or higher is required."
    exit 1
}

Write-Host "  Found: $PYTHON_CMD ($version)"

# ── Step 2: Upgrade pip ──────────────────────────────────────────────────────
Write-Host "Upgrading pip..."

& $PYTHON_CMD -m pip install --upgrade pip

# ── Step 3: Install requirements ─────────────────────────────────────────────
Write-Host "Installing requirements.txt..."

pip install -r requirements.txt

# ── Step 4: Download spaCy model ─────────────────────────────────────────────
Write-Host "Downloading spaCy en_core_web_sm..."

& $PYTHON_CMD -m spacy download en_core_web_sm

# ── Step 5: Create models directory ──────────────────────────────────────────
Write-Host "Creating models/all-MiniLM-L6-v2 directory..."

New-Item -ItemType Directory -Force -Path "models\all-MiniLM-L6-v2" | Out-Null

# ── Step 6: Download ONNX model + tokenizer ─────────────────────────────────
Write-Host "Downloading ONNX model and tokenizer from HuggingFace..."

$pythonScript = @"
import os
import shutil
from huggingface_hub import hf_hub_download

repo_id = "sentence-transformers/all-MiniLM-L6-v2"

os.makedirs("models/all-MiniLM-L6-v2", exist_ok=True)

print("  Downloading model.onnx...")

onnx_path = hf_hub_download(
    repo_id=repo_id,
    filename="onnx/model.onnx",
    local_dir="models/all-MiniLM-L6-v2",
)

nested = os.path.join(
    "models", "all-MiniLM-L6-v2", "onnx", "model.onnx"
)

target = os.path.join(
    "models", "all-MiniLM-L6-v2", "model.onnx"
)

if os.path.isfile(nested) and not os.path.isfile(target):
    shutil.copy2(nested, target)
    print(f"  Moved {nested} -> {target}")

print("  Downloading tokenizer.json...")

tok_path = hf_hub_download(
    repo_id=repo_id,
    filename="tokenizer.json",
    local_dir="models/all-MiniLM-L6-v2",
)

nested_tok = os.path.join(
    "models", "all-MiniLM-L6-v2", "tokenizer.json"
)

if not os.path.isfile(nested_tok):
    shutil.copy2(tok_path, nested_tok)

assert os.path.isfile(target), f"Missing ONNX model at {target}"
assert os.path.isfile(nested_tok), f"Missing tokenizer at {nested_tok}"

print("  Models downloaded successfully.")
"@

& $PYTHON_CMD -c $pythonScript

# ── Step 7: Initialize SQLite schema ─────────────────────────────────────────
Write-Host "Initialising SQLite schema at data/graph.db..."

$sqliteScript = @"
import sqlite3
import os

os.makedirs("data", exist_ok=True)

conn = sqlite3.connect(os.path.join("data", "graph.db"))

conn.execute('''
    CREATE TABLE nodes (
        id TEXT PRIMARY KEY,
        type TEXT,
        content TEXT,
        embedding BLOB,
        priority REAL,
        created_at REAL,
        updated_at REAL,
        access_count INTEGER,
        confidence REAL,
        version INTEGER,
        last_reconciled_version INTEGER
    )
''')

conn.execute('''
    CREATE TABLE edges (
        id TEXT PRIMARY KEY,
        from_node TEXT,
        to_node TEXT,
        type TEXT,
        weight REAL,
        created_at REAL
    )
''')

conn.execute('''
    CREATE TABLE meta (
        key TEXT PRIMARY KEY,
        value TEXT,
        updated_at REAL
    )
''')

conn.commit()
conn.close()

print("  Schema initialised.")
"@

& $PYTHON_CMD -c $sqliteScript

# ── Step 8: Done ────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Setup complete."
Write-Host "Run:"
Write-Host "  conda activate graphdag"
Write-Host "  pytest tests/test_environment.py -v"