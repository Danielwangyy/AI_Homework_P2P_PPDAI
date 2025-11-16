# PowerShell setup script for Windows
# Equivalent to setup.sh but for Windows PowerShell

$ErrorActionPreference = "Stop"

$PROJECT_ROOT = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $PROJECT_ROOT

$VENV_DIR = Join-Path $PROJECT_ROOT ".venv"

# Find Python executable
if ($env:PYTHON) {
    $BASE_PYTHON = $env:PYTHON
} elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    $BASE_PYTHON = "python3"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $BASE_PYTHON = "python"
} else {
    Write-Host "[ERROR] 未找到可用的 Python 解释器，请先安装 Python 后重试。" -ForegroundColor Red
    Write-Host "提示: 可以从 https://www.python.org/downloads/ 下载并安装 Python" -ForegroundColor Yellow
    exit 1
}

# Test if Python works
try {
    $pythonVersion = & $BASE_PYTHON --version 2>&1
    Write-Host "[INFO] 找到 Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python 无法执行: $_" -ForegroundColor Red
    exit 1
}

# Create virtual environment if it doesn't exist
if (-not (Test-Path $VENV_DIR)) {
    Write-Host "[INFO] 未发现 .venv，正在创建虚拟环境..." -ForegroundColor Yellow
    & $BASE_PYTHON -m venv $VENV_DIR
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] 创建虚拟环境失败" -ForegroundColor Red
        exit 1
    }
}

# Determine Python executable in virtual environment
$PYTHON_BIN = Join-Path $VENV_DIR "Scripts\python.exe"
if (-not (Test-Path $PYTHON_BIN)) {
    $PYTHON_BIN = Join-Path $VENV_DIR "bin\python"
    if (-not (Test-Path $PYTHON_BIN)) {
        Write-Host "[ERROR] 找不到虚拟环境中的 Python 可执行文件" -ForegroundColor Red
        exit 1
    }
}

Write-Host "[INFO] 使用虚拟环境: $PYTHON_BIN" -ForegroundColor Green

# Upgrade pip
Write-Host "[INFO] 升级 pip ..." -ForegroundColor Yellow
& $PYTHON_BIN -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] 升级 pip 失败" -ForegroundColor Red
    exit 1
}

# Install requirements
Write-Host ""
Write-Host "[INFO] 安装项目依赖 (environments/requirements.txt)..." -ForegroundColor Yellow
$requirementsFile = Join-Path $PROJECT_ROOT "environments\requirements.txt"
if (-not (Test-Path $requirementsFile)) {
    Write-Host "[ERROR] 找不到 requirements.txt: $requirementsFile" -ForegroundColor Red
    exit 1
}
& $PYTHON_BIN -m pip install -r $requirementsFile
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] 安装依赖失败" -ForegroundColor Red
    exit 1
}

# Install project in editable mode
Write-Host ""
Write-Host "[INFO] 使用 $PYTHON_BIN 安装项目 (editable 模式)..." -ForegroundColor Yellow
& $PYTHON_BIN -m pip install -e .
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] 安装项目失败" -ForegroundColor Red
    exit 1
}

# Verify CLI
Write-Host ""
Write-Host "[INFO] 验证命令帮助..." -ForegroundColor Yellow
$helpOutput = & $PYTHON_BIN -m ai_homework.cli.run_pipeline --help 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host $helpOutput
    Write-Host ""
    Write-Host "[OK] 安装完成，可以开始使用项目啦！" -ForegroundColor Green
    Write-Host ""
    Write-Host "要激活虚拟环境，请运行:" -ForegroundColor Cyan
    Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor Cyan
} else {
    Write-Host $helpOutput -ForegroundColor Red
    Write-Host "[ERROR] 验证失败：未能成功运行 ai_homework CLI。请检查上方输出后重试。" -ForegroundColor Red
    exit 1
}

