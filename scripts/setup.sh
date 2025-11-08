#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "[WARN] 当前未检测到激活的虚拟环境，将直接使用系统 Python。" >&2
  echo "       建议先运行 'source .venv/bin/activate' 或按 README 指引创建并激活虚拟环境。" >&2
fi

PYTHON_BIN="${PYTHON:-python}"

echo "[INFO] 升级 pip ..."
"${PYTHON_BIN}" -m pip install --upgrade pip

echo
echo "[INFO] 安装项目依赖 (environments/requirements.txt)..."
"${PYTHON_BIN}" -m pip install -r environments/requirements.txt

echo
echo "[INFO] 使用 ${PYTHON_BIN} 安装项目 (editable 模式)..."
"${PYTHON_BIN}" -m pip install -e .

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
