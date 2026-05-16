#!/usr/bin/env bash
# ── Graph-DAG Middleware — Environment Setup ──────────────────────────────────
# Run once from the project root: bash scripts/setup_env.sh
# Tested on: Ubuntu 22.04 / 24.04 + Ryzen 5 4600H + GTX 1650 4GB VRAM
# Windows users: run in WSL2 with Ubuntu.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[setup]${NC} $1"; }
warn() { echo -e "${YELLOW}[warn]${NC}  $1"; }
fail() { echo -e "${RED}[fail]${NC}  $1"; exit 1; }

# ── 1. Python version check ───────────────────────────────────────────────────
log "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
major=$(echo "$python_version" | cut -d. -f1)
minor=$(echo "$python_version" | cut -d. -f2)
if [ "$major" -lt 3 ] || { [ "$major" -eq 3 ] && [ "$minor" -lt 10 ]; }; then
    fail "Python 3.10+ required. Found: $python_version"
fi
log "Python $python_version ✓"

# ── 2. Virtual environment ────────────────────────────────────────────────────
log "Creating virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    log "Virtual environment created at .venv/"
else
    warn ".venv already exists — skipping creation"
fi

source .venv/bin/activate
log "Virtual environment activated"

# ── 3. Upgrade pip + install dependencies ────────────────────────────────────
log "Installing dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
log "Dependencies installed ✓"

# ── 4. spaCy language model ───────────────────────────────────────────────────
log "Downloading spaCy en_core_web_sm model..."
python3 -m spacy download en_core_web_sm --quiet
log "spaCy model ready ✓"

# ── 5. ONNX embedding model download ─────────────────────────────────────────
log "Downloading all-MiniLM-L6-v2 ONNX model..."
mkdir -p data/models

python3 - <<'PYEOF'
import os, urllib.request, json

MODEL_DIR = "data/models/all-MiniLM-L6-v2-tokenizer"
ONNX_PATH = "data/models/all-MiniLM-L6-v2.onnx"

os.makedirs(MODEL_DIR, exist_ok=True)

BASE = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main"

# Download ONNX model
if not os.path.exists(ONNX_PATH):
    onnx_url = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"
    print(f"  Downloading ONNX model (~23MB)...")
    urllib.request.urlretrieve(onnx_url, ONNX_PATH)
    print(f"  ONNX model saved to {ONNX_PATH}")
else:
    print(f"  ONNX model already exists at {ONNX_PATH} — skipping")

# Download tokenizer files
tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json"]
for fname in tokenizer_files:
    dest = os.path.join(MODEL_DIR, fname)
    if not os.path.exists(dest):
        url = f"{BASE}/{fname}"
        print(f"  Downloading {fname}...")
        urllib.request.urlretrieve(url, dest)
    else:
        print(f"  {fname} already present — skipping")

print("  Embedding model ready ✓")
PYEOF

# ── 6. SQLite database init ───────────────────────────────────────────────────
log "Initializing SQLite database..."
mkdir -p data/db
python3 - <<'PYEOF'
import sqlite3, os

db_path = "data/db/graph.db"
conn = sqlite3.connect(db_path)
c = conn.cursor()

c.executescript("""
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS nodes (
    id                      TEXT PRIMARY KEY,
    type                    TEXT NOT NULL,
    content                 TEXT NOT NULL,
    embedding               BLOB NOT NULL,
    priority                REAL NOT NULL DEFAULT 0.5,
    created_at              REAL NOT NULL,
    updated_at              REAL NOT NULL,
    access_count            INTEGER NOT NULL DEFAULT 0,
    confidence              REAL NOT NULL DEFAULT 0.8,
    version                 INTEGER NOT NULL DEFAULT 1,
    last_reconciled_version INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS edges (
    id         TEXT PRIMARY KEY,
    from_node  TEXT NOT NULL REFERENCES nodes(id),
    to_node    TEXT NOT NULL REFERENCES nodes(id),
    type       TEXT NOT NULL,
    weight     REAL NOT NULL DEFAULT 0.5,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_node);
CREATE INDEX IF NOT EXISTS idx_edges_to   ON edges(to_node);
CREATE INDEX IF NOT EXISTS idx_nodes_priority ON nodes(priority);
CREATE INDEX IF NOT EXISTS idx_nodes_updated  ON nodes(updated_at);
""")

conn.commit()
conn.close()
print(f"  Database initialized at {db_path} ✓")
PYEOF

# ── 7. Ollama check ───────────────────────────────────────────────────────────
log "Checking Ollama installation..."
if command -v ollama &> /dev/null; then
    log "Ollama found: $(ollama --version 2>/dev/null || echo 'version unknown')"

    log "Checking for llama3.2:3b model (recommended for GTX 1650 4GB VRAM)..."
    if ollama list 2>/dev/null | grep -q "llama3.2:3b"; then
        log "llama3.2:3b already pulled ✓"
    else
        warn "llama3.2:3b not found. Pull it with:"
        warn "  ollama pull llama3.2:3b"
        warn "  (this is the recommended model for your 4GB VRAM)"
        warn "  Alternative: phi3:mini (similar size, different strengths)"
    fi
else
    warn "Ollama not found. Install it:"
    warn "  curl -fsSL https://ollama.com/install.sh | sh"
    warn "Then pull your model:"
    warn "  ollama pull llama3.2:3b"
fi

# ── 8. .env file ─────────────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    log "Creating .env from template..."
    cat > .env <<'EOF'
# Graph-DAG Middleware — Environment Variables
OLLAMA_BASE_URL=http://localhost:11434
MIDDLEWARE_PORT=8080
LOG_LEVEL=info
CONFIG_PATH=config/config.yaml
EOF
    log ".env created ✓"
else
    warn ".env already exists — skipping"
fi

# ── 9. Run quick sanity check ─────────────────────────────────────────────────
log "Running sanity checks..."
python3 - <<'PYEOF'
import sys

errors = []

try:
    import fastapi
    print(f"  fastapi {fastapi.__version__} ✓")
except ImportError as e:
    errors.append(f"fastapi: {e}")

try:
    import hnswlib
    print(f"  hnswlib ✓")
except ImportError as e:
    errors.append(f"hnswlib: {e}")

try:
    import onnxruntime as ort
    print(f"  onnxruntime {ort.__version__} ✓")
except ImportError as e:
    errors.append(f"onnxruntime: {e}")

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    print(f"  spacy {spacy.__version__} + en_core_web_sm ✓")
except Exception as e:
    errors.append(f"spacy: {e}")

try:
    import numpy as np
    print(f"  numpy {np.__version__} ✓")
except ImportError as e:
    errors.append(f"numpy: {e}")

try:
    import sqlite3
    print(f"  sqlite3 {sqlite3.sqlite_version} ✓")
except ImportError as e:
    errors.append(f"sqlite3: {e}")

try:
    import apscheduler
    print(f"  apscheduler ✓")
except ImportError as e:
    errors.append(f"apscheduler: {e}")

if errors:
    print("\nErrors:")
    for err in errors:
        print(f"  ✗ {err}")
    sys.exit(1)
else:
    print("\n  All imports OK ✓")
PYEOF

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
log "═══════════════════════════════════════════════════════"
log "  Setup complete."
log ""
log "  To activate the environment in future sessions:"
log "    source .venv/bin/activate"
log ""
log "  To start the middleware proxy:"
log "    python3 -m src.proxy.server"
log ""
log "  To run tests:"
log "    pytest tests/ -v"
log ""
log "  Recommended LLM for your hardware (GTX 1650 4GB):"
log "    ollama pull llama3.2:3b    (~2.0 GB VRAM)"
log "    ollama pull phi3:mini      (~2.2 GB VRAM)"
log "    ✗ DO NOT use 7B+ models — they will OOM on 4GB VRAM"
log "═══════════════════════════════════════════════════════"
