# 2025-11 目录与包结构重构迁移指南

本次重构旨在支持 Git 协作与模块化开发。以下列出主要调整及使用建议，协助团队顺利迁移。

## 1. 源码布局

| 调整前 | 调整后 | 说明 |
| --- | --- | --- |
| `src/data/`、`src/features/` 等散落模块 | `src/ai_homework/{data,features,models,...}` | 采用单一包 `ai_homework`，统一导入路径，方便打包与测试。 |
| `scripts/run_full_pipeline.py` 直接调用 `src/pipelines` | 通过 Agent 说“请执行 python -m ai_homework.cli.run_pipeline” | 新增 CLI 封装（原脚本已移除），支持 `--skip-data/--skip-train` 等参数。 |
| `src/utils/metrics.py` | `src/ai_homework/evaluation/metrics.py` | 指标与可视化函数迁移至 `evaluation` 子包。 |

迁移步骤：
1. 更新代码中的导入语句：`from src.xxx` 替换为 `from ai_homework.xxx`（已在仓库完成，若自有分支需同步）。
2. 如需运行某个模块，请在 Agent 模式下说明：“请在当前目录执行 python -m ai_homework.<module>”，Agent 会自动切换并执行（注意将 `<module>` 替换为真实路径，如 `pipelines.prepare_data`）。

## 2. 产出物与数据目录

| 调整前 | 调整后 | 说明 |
| --- | --- | --- |
| `artifacts/`、`models/`、`logs/` 等散落在根目录 | `outputs/{artifacts,models,experiments,logs,reports}` | 统一纳入 `outputs/`（默认 `.gitignore`），便于清理与备份。 |
| `reports/figures`、`reports/tables` | `outputs/reports/figures`、`outputs/reports/tables` | 自动生成的图表与表格归档至 `outputs`。 |
| `environments/p2p_env` 虚拟环境 | 推荐使用 `.venv/` | 不再保留本地虚拟环境目录，转为 README 指南。 |

配置更新：
- `configs/data_processing.yaml` 与 `configs/model_training.yaml` 中的路径已改为相对路径，并指向新的 `outputs/` 结构。
- 使用自定义路径时，可在配置文件写入相对或绝对路径；流水线会自动解析为绝对路径。

## 3. 文档与测试

- 新增 `README.md`、`CONTRIBUTING.md`，说明目录结构、常用命令与协作流程。
- `docs/testing_strategy.md` 已同步更新，所有命令改为 `python -m ai_homework...` 格式，并明确测试产物路径。
- `tests/` 目录将逐步按包结构拆分（示例：`tests/configs/test_configs.py`），请将新的测试文件放入对应子目录。

## 4. 后续注意事项

- 执行流水线前，请确认相应目录存在或由脚本自动创建。
- 若旧分支仍指向 `artifacts/` 等老路径，请合并 `main` 后重新运行流程，避免输出路径不一致。
- CI / 自动化脚本可通过 Agent 或脚本调用 `python -m ai_homework.cli.run_pipeline`，搭建环境时同样可以让 Agent 执行 `pip install -r environments/requirements.txt`。

如在迁移过程中遇到问题，可参考 `docs/project_structure.md` 或提交 Issue 与维护者沟通。

