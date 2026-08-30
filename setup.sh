#!/usr/bin/env bash
# SecureStegVault v3.2 — one-time setup
# Installs Node deps + Python venv (PEP 668 / Kali / Debian safe)
# Includes research-comparison dependencies (scipy, scikit-image, psutil, opencv).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${CYAN}[setup]${NC} $*"; }
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[ERR]${NC} $*" >&2; }

echo "========================================="
echo "  SecureStegVault v3.2 — Setup"
echo "  Research comparison + CNN EMD-OPAP"
echo "========================================="

# ── Node ─────────────────────────────────────────────────────────────────────
log "[1/3] Installing Node.js dependencies..."
if ! command -v npm >/dev/null 2>&1; then
  err "npm not found. Install Node.js first (e.g. apt install nodejs npm)."
  exit 1
fi
npm install
ok "Node dependencies ready."

# ── Python venv ──────────────────────────────────────────────────────────────
log "[2/3] Creating Python virtual environment (.venv)..."
if ! command -v python3 >/dev/null 2>&1; then
  err "python3 not found."
  exit 1
fi

if ! python3 -c "import venv" 2>/dev/null; then
  warn "python3-venv not available."
  warn "On Kali/Debian run:  sudo apt install -y python3-venv python3-full"
fi

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
  ok "Created .venv"
else
  ok ".venv already exists"
fi

# shellcheck disable=SC1091
source .venv/bin/activate

log "[3/3] Installing Python dependencies into .venv..."
python -m pip install --upgrade pip wheel setuptools >/dev/null
python -m pip install -r requirements.txt

# Optional extras (best-effort)
python -m pip install argon2-cffi 2>/dev/null || warn "argon2-cffi optional — skipped"
python -m pip install pytest 2>/dev/null || true

ok "Python packages installed in .venv"

# ── Dirs used by comparison / experiments ────────────────────────────────────
mkdir -p models/paper1/official models/paper2/official models/paper3/official
mkdir -p datasets/covers experiments .pids
ok "models/ datasets/ experiments/ directories ready"

# ── Make launchers executable ────────────────────────────────────────────────
chmod +x start.sh setup.sh 2>/dev/null || true

echo
echo "========================================="
echo " Setup complete!"
echo
echo "  Start (dev):   ./start.sh"
echo "  Start (prod):  ./start.sh prod"
echo "  Python only:   ./start.sh python"
echo "  Stop:          ./start.sh stop"
echo "  Tests:         source .venv/bin/activate && PYTHONPATH=. pytest backend/comparison/tests -q"
echo
echo "  UI:  http://localhost:3000"
echo "  API: http://127.0.0.1:8001"
echo
echo "  Comparison tab → Live benchmark + Model status"
echo "  Optional Paper1 weights → models/paper1/official/*.pth"
echo "========================================="
