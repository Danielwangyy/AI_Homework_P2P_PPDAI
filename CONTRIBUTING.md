# 贡献指南

感谢你愿意为 P2P 借款风控项目贡献力量！为确保团队协作顺畅，请遵循以下约定。

## 开发流程

1. **准备环境**
   - 按照 `README.md` 中的说明创建虚拟环境并安装依赖。
   - 确认 `python -m pytest` 可正常执行。
2. **创建分支**
   - 从 `main`（或指定主干分支）拉取最新代码。
   - 使用语义化分支命名，例如 `feature/feature-engineering-updates`、`fix/pipeline-scaler-bug`。
3. **编码规范**
   - 代码置于 `src/ai_homework/` 包内，遵循现有模块划分。
   - 保持函数/类文档字符串，必要时补充类型注解。
   - 所有运行产物（模型、日志、图表等）应写入 `outputs/`，避免污染仓库。
4. **测试与验证**
   - 本地运行 `python -m pytest`，确保新增/修改逻辑被覆盖。
   - 若引入流水线变更，请至少在缩略数据集上执行 `python -m ai_homework.cli.run_pipeline --skip-train` 或 `--skip-data` 进行冒烟验证。
5. **提交与合并**
   - 使用明晰的提交信息，例如 `feat(pipelines): add cli wrapper for run_pipeline`。
   - Pull Request 需包含：变更摘要、测试结果、潜在风险与回滚方案。

## 文档与配置

- 若更新目录或配置结构，请同步维护：
  - `docs/project_structure.md`
  - `docs/migration/`（新增迁移指南，描述旧/新差异）
  - `docs/testing_strategy.md`（补充测试策略变动）
- 在 `README.md` 中补充新的运行方式或依赖。

## 依赖管理

- 所有 Python 依赖写入 `environments/requirements.txt`。
- 新增依赖需在 PR 中说明用途及兼容性考虑。
- 若需要固定版本，可在审查后同步生成 `requirements.lock`（可选）。

## 代码审查要点

- **可维护性**：模块边界清晰，避免耦合；配置使用 YAML，并通过 `ai_homework.utils.config.load_yaml` 读取。
- **数据安全**：敏感数据禁止入库；确保输出路径指向 `outputs/`。
- **性能与稳定性**：审视循环、数据量敏感操作，必要时提供参数化控制。
- **测试覆盖**：新增或修改的核心逻辑应附带单元测试；端到端流程变更需补充集成测试。

如有任何问题或建议，可在 Issue 中讨论或与项目维护者直接联系。期待你的贡献！ 🎉

