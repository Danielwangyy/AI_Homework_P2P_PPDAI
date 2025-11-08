#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON:-python}"

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
