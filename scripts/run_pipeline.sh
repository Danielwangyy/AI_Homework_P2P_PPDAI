#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
elif [[ -x "$PROJECT_ROOT/.venv/Scripts/python.exe" ]]; then
  PYTHON_BIN="$PROJECT_ROOT/.venv/Scripts/python.exe"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "[ERROR] 未找到可用的 Python 解释器，请先安装 Python 或激活虚拟环境后再试。" >&2
  exit 127
fi

"${PYTHON_BIN}" - <<'PY'
import importlib
import sys

try:
    importlib.import_module("ai_homework")
except ImportError:
    sys.stderr.write("[ERROR] 未能导入 ai_homework，请先运行 ./scripts/setup.sh 完成安装。\n")
    sys.exit(1)
PY

exec "${PYTHON_BIN}" -m ai_homework.cli.run_pipeline "$@"
