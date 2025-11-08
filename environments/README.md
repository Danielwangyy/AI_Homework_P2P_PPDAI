# 虚拟环境使用说明

## 1. 创建与激活

推荐在项目根目录下使用 `.venv/` 作为本地虚拟环境目录（默认被 `.gitignore` 忽略）。

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# 或
.venv\Scripts\Activate.ps1       # Windows PowerShell
```

> 也可以使用 Conda / Mamba 创建环境，保持 Python 版本 ≥3.9。

## 2. 安装依赖

```bash
pip install --upgrade pip
pip install -r environments/requirements.txt
```

安装完成后可使用 `pip list` 或 `pip freeze > environments/requirements.lock` 查看依赖。
当 `requirements.txt` 更新时（例如新增 LightGBM、CatBoost、SHAP、PyTest 等库），请重新执行上述命令，确保训练与测试脚本具备完整依赖。

## 3. 退出环境

```bash
deactivate
```

## 4. 其他说明
- `environments/` 目录仅存放依赖说明文件，请勿将本地虚拟环境提交到 Git。
- 项目默认依赖 `xgboost`、`lightgbm`、`catboost`、`shap` 以及 `pytest`/`pytest-cov` 等测试工具；运行 `python -m ai_homework.cli.run_pipeline`或 `python -m pytest` 前需确认它们已安装成功。
- 如需 GPU 加速，请根据设备情况额外安装相应版本的 `xgboost`（或相关深度学习框架），并在配置文件中调整参数。
- 在 Notebook 或批处理任务启动前先激活虚拟环境，确保依赖版本一致可复现。

