#!/usr/bin/env bash
# SecureStegVault launcher
# Usage:
#   ./start.sh          → development mode (Vite HMR + Python CNN backend)
#   ./start.sh prod     → production mode (built assets + Node server)
#   ./start.sh python   → Python FastAPI only (port 8001)
#   ./start.sh stop     → stop background processes started by this script

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODE="${1:-dev}"
PID_DIR="$ROOT/.pids"
mkdir -p "$PID_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${CYAN}[SecureStegVault]${NC} $*"; }
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[ERR]${NC} $*" >&2; }

# Prefer project venv (created by setup.sh) — required on Kali/Debian PEP 668
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
  export PATH="$ROOT/.venv/bin:$PATH"
else
  PY="python3"
fi

# ── helpers ──────────────────────────────────────────────────────────────────

is_port_free() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ! ss -ltn "sport = :$port" 2>/dev/null | grep -q ":$port"
  elif command -v lsof >/dev/null 2>&1; then
    ! lsof -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
  else
    return 0
  fi
}

wait_http() {
  local url="$1" timeout="${2:-40}" i=0
  while (( i < timeout )); do
    if curl -sf "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
    ((i++)) || true
  done
  return 1
}

stop_pidfile() {
  local name="$1" file="$PID_DIR/$name.pid"
  if [[ -f "$file" ]]; then
    local pid
    pid="$(cat "$file" 2>/dev/null || true)"
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      log "Stopping $name (pid $pid)…"
      kill "$pid" 2>/dev/null || true
      sleep 0.4
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$file"
  fi
}

cleanup_on_exit() {
  if [[ "${KEEP_RUNNING:-0}" != "1" ]]; then
    stop_pidfile python
  fi
}
trap cleanup_on_exit EXIT INT TERM

# ── stop ─────────────────────────────────────────────────────────────────────

if [[ "$MODE" == "stop" ]]; then
  log "Stopping SecureStegVault services…"
  stop_pidfile python
  stop_pidfile node
  for port in 8001 3000 5173; do
    if command -v lsof >/dev/null 2>&1; then
      pids=$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
      if [[ -n "${pids:-}" ]]; then
        warn "Killing leftover process(es) on :$port → $pids"
        # shellcheck disable=SC2086
        echo "$pids" | xargs -r kill 2>/dev/null || true
      fi
    fi
  done
  ok "Stopped."
  exit 0
fi

# ── dependency checks ────────────────────────────────────────────────────────

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    err "Required command not found: $1"
    err "Run:  bash setup.sh"
    exit 1
  fi
}

need_cmd node
need_cmd npm

if [[ ! -d node_modules ]]; then
  warn "node_modules missing — running npm install…"
  npm install
fi

if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  warn "No .venv found. On Kali/Debian you should run:  bash setup.sh"
  warn "Continuing with system python3 (may fail if packages are missing)."
fi

# ── start Python FastAPI (CNN backend) ───────────────────────────────────────

start_python() {
  if ! is_port_free 8001; then
    warn "Port 8001 already in use — assuming Python backend is already running."
    return 0
  fi

  log "Starting Python CNN backend (uvicorn :8001)…"
  log "  interpreter: $PY"

  (
    export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
    exec "$PY" -m uvicorn backend.main:app \
      --host 127.0.0.1 \
      --port 8001 \
      --log-level warning
  ) >"$PID_DIR/python.log" 2>&1 &
  echo $! >"$PID_DIR/python.pid"

  if wait_http "http://127.0.0.1:8001/api/health" 40; then
    ok "Python backend ready → http://127.0.0.1:8001"
  else
    warn "Python backend did not respond in time."
    warn "Check log: $PID_DIR/python.log"
    warn "Frontend will fall back to pure TypeScript pipeline."
    if [[ -f "$PID_DIR/python.log" ]]; then
      echo "----- last 20 lines of python.log -----"
      tail -20 "$PID_DIR/python.log" || true
      echo "---------------------------------------"
    fi
  fi
}

# ── modes ────────────────────────────────────────────────────────────────────

case "$MODE" in
  python)
    KEEP_RUNNING=1
    start_python
    log "Python-only mode. Logs: $PID_DIR/python.log"
    log "Press Ctrl+C to stop."
    wait "$(cat "$PID_DIR/python.pid")"
    ;;

  prod)
    if [[ ! -f dist/server.cjs ]]; then
      log "Production build not found — building…"
      npm run build
    fi
    start_python
    KEEP_RUNNING=1
    log "Starting production Node server…"
    exec node dist/server.cjs
    ;;

  dev|*)
    start_python
    KEEP_RUNNING=1
    log "Starting development server (tsx server.ts)…"
    log "  UI + API will be available shortly (default http://localhost:3000)"
    log "  Python CNN backend: http://127.0.0.1:8001"
    log "  New pages: Benchmark · Compare (in the top nav)"
    echo
    exec npm run dev
    ;;
esac
