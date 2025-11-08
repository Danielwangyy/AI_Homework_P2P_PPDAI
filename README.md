# AI Homework P2P Loan Default Project

本项目聚焦拍拍贷（PPDAI）借款违约风险建模，覆盖数据清洗、特征工程、模型训练与结果可视化。为了帮助完全不会写代码的同学，我们把所有操作都写成“对 Cursor Agent 说的一句话”。你只需切换到 Agent 模式，用自然语言照着说即可。

---

## 快速开始（零基础版）

1. **Fork 仓库**  
   登录 GitHub，访问 `https://github.com/Danielwangyy/AI_Homework_P2P_PPDAI`，点击右上角 `Fork` → `Create fork`。  

2. **让 Agent 克隆仓库**  
   - 在自己的仓库页面复制 `SSH` 或 `HTTPS` 地址。  
   - 回到 Cursor，确认聊天窗口处于 `Agent` 模式，然后说：  
     ```
     请把仓库克隆到 ~/Documents，仓库地址是 <粘贴你的仓库链接>
     ```  
   - 克隆完成后继续说：`请进入 ~/Documents/AI_Homework_P2P_PPDAI，并告诉我当前路径`。
   - Cursor 工作区选择 `File → Open Workspace...` 打开该目录。

3. **创建 Python 虚拟环境并安装依赖**  
   依次对 Agent 说：
   - `请使用 python3 在当前目录创建名为 .venv 的虚拟环境`
   - `请帮我激活这个虚拟环境`（macOS/Linux 会执行 `source .venv/bin/activate`，Windows 会执行 `.venv\Scripts\activate`）
   - `请先升级 pip`
   - `请安装 environments/requirements.txt 里的依赖`

4. **准备数据**  
   把老师提供的 `P2P_PPDAI_DATA` 资料夹放入项目的 `data/raw/source_data/`（如没有该目录，可让 Agent 创建：`请在 data/raw/ 下创建 source_data 文件夹`）。需至少包含 `LC.csv`、`LP.csv`、`LCIS.csv` 及配套字典。

5. **验证环境可用**  
   让 Agent 运行：`请执行 python -m ai_homework.cli.run_pipeline --help`。看到参数说明代表环境与依赖安装成功。

完成以上 5 步，你就拥有可运行的本地环境和数据副本了。

---

## 经常会用到的 Agent 话术

- **运行完整流水线（数据准备 + 模型训练）**  
  `请执行 python -m ai_homework.cli.run_pipeline`

- **只做数据准备**  
  `请执行 python -m ai_homework.pipelines.prepare_data --config configs/data_processing.yaml`

- **只训练模型**  
  `请执行 python -m ai_homework.pipelines.train_models --config configs/model_training.yaml`

- **跑测试**  
  `请执行 python -m pytest`

- **查看日志文件**  
  `请把 outputs/logs/data_pipeline.log 的内容展示给我`

如果 Agent 询问“需要进入哪个目录”，请回答：`进入 ~/Documents/AI_Homework_P2P_PPDAI`（或你实际存放项目的位置）。

---

## 协作流程（简版）

1. **确认当前分支**  
   `请告诉我当前 Git 分支`

2. **新建工作分支**  
   `请创建并切换到分支 feature/<你的主题>`

3. **编辑文件并保存**  
   在 Cursor 编辑器中直接改内容，保存即可。

4. **查看改动**  
   `请告诉我这一次修改包含哪些文件`

5. **提交代码**  
   `请把所有改动加入暂存区并使用 “Update xxx” 作为提交说明`

6. **推送到远程**  
   `请把当前分支推送到我的 GitHub 仓库`

7. **发起 PR**  
   打开 GitHub → 你的仓库 → 点击 “Compare & pull request” → 填写说明并提交。

详细图文提示请参考 `docs/collaboration_guide.md`。

---

## 目录速览

- `src/ai_homework/`：核心代码（数据、特征、模型、流水线、评估等模块）。  
- `configs/`：所有 YAML 配置文件。  
- `data/`：原始与加工数据（默认不纳入 Git）。  
- `models/`、`reports/`、`logs/`：流水线生成的模型成果与可视化。  
- `docs/`：项目文档；特别推荐先读 `collaboration_guide.md`、`project_structure.md`。  
- `tests/`：单元/集成测试，对应 `src/` 结构。  

如需了解更详细的文件组织，可让 Agent 打开 `docs/project_structure.md` 并朗读重点。

---

## 面向教学场景的建议

- 运行模型前，先让 Agent 帮忙备份数据或拷贝样本，以免误删原始文件。  
- 每次实验结论最好写入 `reports/summary.md` 或新增文档，方便团队复现。  
- 调整配置时（如 `configs/model_training.yaml`），记得在 PR 描述里说明改动动机。  
- 新增 Python 依赖？先让 Agent 编辑 `environments/requirements.txt`，再运行 `pip install -r environments/requirements.txt`。

---

## 遇到问题怎么办？

- **环境报错**  
  把错误信息贴给 Agent，询问：“请解释这段报错并告诉我怎么修复。”  
- **Git 冲突**  
  让 Agent 查看冲突文件，然后在 Cursor 中手动合并，最后说：“请继续完成合并并提交。”  
- **不知道下一步做什么**  
  打开 `docs/collaboration_guide.md` 或 `docs/project_summary.md`，请 Agent 总结关键步骤。  
- **需要帮助**  
  在项目的 GitHub 仓库里开 Issue，或在团队群里描述情况并附上 Agent 的输出。

---

我们相信“让 Agent 做命令，自己专注思考”能让你把时间花在理解机器学习任务上，而不是纠结终端操作。祝学习顺利，也欢迎随时在 PR、Issue 中交流想法！

