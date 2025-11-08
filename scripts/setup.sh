#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

VENV_DIR="$PROJECT_ROOT/.venv"

if [[ -n "${PYTHON:-}" ]]; then
  BASE_PYTHON="${PYTHON}"
elif command -v python3 >/dev/null 2>&1; then
  BASE_PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
  BASE_PYTHON="python"
else
  echo "[ERROR] 未找到可用的 Python 解释器，请先安装 Python 后重试。" >&2
  exit 127
fi

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[INFO] 未发现 .venv，正在创建虚拟环境..."
  "$BASE_PYTHON" -m venv "$VENV_DIR"
fi

if [[ -f "$VENV_DIR/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
elif [[ -f "$VENV_DIR/Scripts/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/Scripts/activate"
else
  echo "[ERROR] 找不到虚拟环境的激活脚本，请删除 .venv 后重试。" >&2
  exit 1
fi

if [[ -x "$VENV_DIR/bin/python" ]]; then
  PYTHON_BIN="$VENV_DIR/bin/python"
elif [[ -x "$VENV_DIR/Scripts/python.exe" ]]; then
  PYTHON_BIN="$VENV_DIR/Scripts/python.exe"
else
  PYTHON_BIN="$(command -v python)"
fi

echo "[INFO] 已在脚本内激活虚拟环境 (.venv)。"

echo "[INFO] 升级 pip ..."
"$PYTHON_BIN" -m pip install --upgrade pip

echo
echo "[INFO] 安装项目依赖 (environments/requirements.txt)..."
"$PYTHON_BIN" -m pip install -r environments/requirements.txt

echo
echo "[INFO] 使用 $PYTHON_BIN 安装项目 (editable 模式)..."
"$PYTHON_BIN" -m pip install -e .

echo
echo "[INFO] 验证命令帮助..."
if "${PYTHON_BIN}" -m ai_homework.cli.run_pipeline --help >/tmp/ai_homework_setup_help.log 2>&1; then
  cat /tmp/ai_homework_setup_help.log
  echo
  echo "[OK] 安装完成，可以开始使用项目啦！"
else
  cat /tmp/ai_homework_setup_help.log >&2
  echo "[ERROR] 验证失败：未能成功运行 ai_homework CLI。请检查上方输出后重试。" >&2
  exit 1
fi
