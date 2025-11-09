# AI Homework P2P Loan Default Project

本仓库是拍拍贷借款违约预测课程作业的基线实现。我们把所有操作拆成一句句“对 Cursor Agent 说的话”，即使从没写过代码，也能按顺序完成作业。

---

## 0. 你将完成什么？

1. 下载项目代码并放到自己的电脑上  
2. 安装运行所需的 Python 环境  
3. 放好老师提供的原始数据  
4. 运行数据准备和模型训练流程，生成结果  
5. （选做）把自己的改动提交到 GitHub 或压缩后上交

---

## 1. 准备工作

| 名称 | 必需 | 说明 |
| --- | --- | --- |
| 电脑 (macOS / Windows / Linux) | ✅ | 需保证有至少 20GB 空间 |
| Cursor 编辑器 | ✅ | 从 <https://cursor.sh/> 下载并安装 |
| Git | ✅ | macOS 通常已内置；Windows 可从 <https://git-scm.com/download/win> 安装 |
| 老师提供的数据包 `P2P_PPDAI_DATA` | ✅ | 内含 `LC.csv`、`LP.csv`、`LCIS.csv` 等 |
| GitHub 账号 | 可选 | 没账号也可以学习；若要提交 PR，建议注册 |

> 不会安装 Git？打开 Cursor 的 Agent 聊天，直接说：`请帮我安装 git`，按提示完成即可。

---

## 2. 获取项目代码

### 方案 A：没有 GitHub 账号

1. 打开 Cursor，确认聊天窗口上方显示 `Agent`。若是 `Ask`，点一下切换。
2. 依次对 Agent 说：
   ```
   请把仓库克隆到 ~/Documents，仓库地址是 https://github.com/Danielwangyy/AI_Homework_P2P_PPDAI.git
   ```
   ```
   请进入 ~/Documents/AI_Homework_P2P_PPDAI，并告诉我当前路径
   ```
3. 在 Cursor 菜单选择 `File → Open Workspace...`，打开同一个目录。

> 没权限克隆？也可以在浏览器打开仓库页面，点击 `Code → Download ZIP`，解压到 `~/Documents/AI_Homework_P2P_PPDAI`，再让 Agent 进入该目录。

### 方案 B：想把作业提交到 GitHub

1. 注册 GitHub（参考 `docs/collaboration_guide.md` 第②步）。
2. 在浏览器访问原始仓库 → 点击 `Fork` → `Create fork`。
3. 复制自己仓库的地址（例如 `https://github.com/你的用户名/AI_Homework_P2P_PPDAI.git`）。
4. 依次对 Agent 说：
   ```
   请把仓库克隆到 ~/Documents，仓库地址是 <粘贴你的仓库地址>
   ```
   ```
   请进入 ~/Documents/AI_Homework_P2P_PPDAI
   ```

---

## 3. 先让 Cursor Agent 熟悉环境

每次执行命令前确保 Agent 真的在项目根目录。可以随时说：
```
请告诉我当前路径
```
若路径不是 `~/Documents/AI_Homework_P2P_PPDAI`（或你指定的目录），说：
```
请进入 ~/Documents/AI_Homework_P2P_PPDAI
```

---

## 4. 安装项目依赖（Agent 版脚本）

**推荐第一句话：**
```
请执行 ./scripts/setup.sh
```

- 这个脚本会在项目根目录自动运行 `python -m pip install -e .`，并顺带执行 `python -m ai_homework.cli.run_pipeline --help`，只要能看到参数说明，就代表环境就绪。
- 如果当前目录没有 `.venv`，脚本会自动创建一个虚拟环境、在内部安装依赖并完成验证；执行完成后，可手动运行 `source .venv/bin/activate` 保持后续会话使用该环境。
- 如果脚本提示“未检测到激活的虚拟环境”，按照提示先创建并激活 `.venv`，再运行一次脚本即可。
- Windows 用户可在 Git Bash、WSL 或任何能执行 `.sh` 的终端里运行；如果终端提示找不到 `pip`，脚本会一起把报错输出展示出来，方便排查。

> 想了解脚本里做了什么？打开 `scripts/setup.sh` 看注释即可。

### 备用方案：一步步手动执行（用于排障或进阶学习）

1. `请使用 python3 在当前目录创建名为 .venv 的虚拟环境`
2. `请激活当前目录下的 .venv 虚拟环境`
3. `请升级 pip`
4. `请安装 environments/requirements.txt 里的依赖`
5. `请执行 python -m pip install -e .`
6. `请执行 python -m ai_homework.cli.run_pipeline --help`

手动执行时，只要最后一步能看到参数介绍，同样表示环境无误。

---

## 5. 放置原始数据

1. 在文件管理器中打开仓库根目录的 `data/raw/`。
2. 把老师发的 `P2P_PPDAI_DATA` 整个文件夹拷贝进去，并重命名为 `source_data`（最终路径：`data/raw/source_data/`）。
3. 目录中至少要包含：
   - `LC.csv`
   - `LP.csv`
   - `LCIS.csv`
   - `LCLP数据字典.xlsx`
   - `LCIS数据字典.xlsx`
   - 其余说明文件可一并保留

> 如果目录不存在，让 Agent 创建：`请在 data/raw/ 下新建 source_data 目录`。

---

## 6. 运行项目（最常用的几句话）

| 目标 | 直接对 Agent 说 |
| --- | --- |
| 初始化环境（推荐） | `请执行 ./scripts/setup.sh` |
| 跑完整流程（推荐脚本） | `请执行 ./scripts/run_pipeline.sh` |
| 跑完整流程（直接使用 python） | `请执行 python -m ai_homework.cli.run_pipeline` |
| 只准备数据 | `请执行 python -m ai_homework.cli.run_pipeline --skip-train` |
| 只训练模型 | `请执行 python -m ai_homework.cli.run_pipeline --skip-data` |
| 查看数据准备日志 | `请把 outputs/logs/data_preparation.log 展示给我` |
| 查看模型训练日志 | `请把 outputs/logs/model_training.log 展示给我` |
| 运行测试 | `请执行 python -m pytest` |

> `./scripts/run_pipeline.sh` 会先尝试导入 `ai_homework`，若失败会提醒你先运行 `./scripts/setup.sh`。需要传参时直接写在脚本后面即可，例如 `./scripts/run_pipeline.sh --skip-data`。

运行成功后，生成的模型、图表、表格会出现在 `outputs/` 目录下。

---

## 7. 提交作业的方式

### 方案 A：压缩提交（无 GitHub 账号）

1. 运行完所有流程后，在文件管理器中右键项目根目录 → “压缩/打包”。  
2. 把压缩包发送给老师即可（记得数据体积太大时，可只保留 `outputs/`、`docs/`、`configs/`、`src/` 等关键内容）。

### 方案 B：提交到自己的 GitHub 仓库

按照下列顺序依次对 Agent 说：
```
请告诉我当前的改动            # 查看状态
请把所有改动加入暂存区
请用说明 "完成数据与模型运行" 创建一次提交
请把当前分支推送到我的 GitHub 仓库
```
然后在 GitHub 网页点击 “Compare & pull request”，填写说明并提交 PR。

更完整的协作流程请看 `docs/collaboration_guide.md`。

---

## 8. 常见疑问速查

- **Agent 提示没有 git/python？**  
  直接让 Agent 安装，或按提示在终端运行对应安装命令。

- **Agent 说权限不足 / 无法进入目录？**  
  先让它执行 `pwd` 确认路径，再 `ls` 看看当前有哪些文件；必要时重新进入项目根目录。

- **运行时报错**  
  把报错复制进聊天，问：“请解释这段报错并告诉我怎么修复。”

- **想了解项目结构与角色说明**  
  让 Agent 打开并朗读 `docs/project_structure.md`、`docs/project_summary.md`。
- **想搞清楚文档阅读顺序**  
  查看 `docs/reading_guide.md`，按推荐路径循序渐进地阅读。
- **想对照函数级实现细节**  
  参考 `docs/beginner_function_walkthrough.md`，了解每个步骤背后调用了哪些脚本与函数。

- **没时间每次开口？**  
  可以把常用命令复制成备忘录，直接粘贴给 Agent。

---

## 9. 目录速览（查阅更多内容）

- `src/ai_homework/`：核心代码（数据处理、模型训练、评估工具等）  
- `configs/`：YAML 配置文件，可修改参数  
- `data/`：原始数据、临时数据、最终数据（已默认忽略原始数据内容）  
- `outputs/`：程序一键生成的模型、报告与日志  
- `docs/`：文档与操作指南（重点推荐 `collaboration_guide.md`）  
- `tests/`：单元测试/集成测试样例  

---

## 10. 遇到困难怎么办？

1. 先让 Agent 描述错误原因与修复建议。  
2. 查看 `outputs/logs/` 里的日志文件，了解运行阶段。  
3. 查阅 `docs/` 中的说明文档。  
4. 仍未解决？请在课程讨论区或 GitHub Issue 中贴出报错与日志。

---

祝学习顺利！放心把重复的命令交给 Cursor Agent，自己专注理解数据与模型即可。 若对流程有改进建议，欢迎在 `docs/` 中补充，或发起 Issue/PR 与同学交流。***

