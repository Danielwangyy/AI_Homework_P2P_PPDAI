# 协作指南：从零开始加入项目

目标：让完全不会写代码的同学，仅凭 Cursor 的 **Agent 模式** 和自然语言交流，就能完成本项目的克隆、修改与提交。每一步都写成“告诉 Agent 要做什么”，照着说即可。

---

## ① 认识 Cursor Agent

1. 打开 Cursor，点击右上角聊天窗口上方的 `Agent`。如果显示的是 `Ask`，点一下即可切换为 `Agent`。  
2. 之后所有需要动手的事情，都通过一句自然语言告诉 Agent。例如：  
   - `请帮我运行 git --version`  
   - `请把仓库克隆到 ~/Documents`  
   - `请查看当前分支`  
3. Agent 会在后台自动执行命令，并把结果回复给你。你无需自己打开终端。

> 小贴士：如果 Agent 提示“切换到目标目录”，继续对它说“请进入 ~/Documents/AI_Homework_P2P_PPDAI”即可。

---

## ② 注册 GitHub 账号

1. 用浏览器访问 <https://github.com/>。
2. 点击右上角 `Sign up`，填写邮箱、密码和用户名。
3. 完成邮箱验证并成功登录。
4. 记住自己的 GitHub 用户名与密码，后续会用到。

---

## ③ 确认或安装 Git

1. 回到 Cursor Agent 聊天窗口，对 Agent 说：`请帮我运行 git --version`。  
2. 看到回复里包含 `git version 2.x.x` 说明 Git 已安装；否则按系统提示安装：
   - **macOS**：系统会自动弹出安装 Xcode Command Line Tools，点击“安装”并等待完成，然后再让 Agent 运行一次 `git --version`。
   - **Windows**：打开浏览器访问 <https://git-scm.com/download/win> 按默认选项安装。安装好后，再回到 Agent 说：`请帮我运行 git --version`。
   - **Linux**：对 Agent 说 `请帮我运行 sudo apt-get install git`（Ubuntu/Debian）或 `请帮我运行 sudo dnf install git`（Fedora）。完成后再检查 `git --version`。

---

## ④ 设置 Git 个人信息

在 Agent 窗口依次告诉它：

```
请帮我运行 git config --global user.name "你的GitHub用户名"
```
```
请帮我运行 git config --global user.email "注册GitHub时使用的邮箱"
```

然后再说：`请帮我运行 git config --global --list`，确认信息被保存。如果暂时不想配置，也可以在之后需要提交时再补。

> 想使用 SSH 免密登录？流程同样可以通过 Agent 完成：  
> 1. `请帮我运行 ssh-keygen -t ed25519 -C "你的邮箱"`  
> 2. `请帮我运行 cat ~/.ssh/id_ed25519.pub` 并复制输出  
> 3. 登录 GitHub → `Settings` → `SSH and GPG keys` → `New SSH key` → 粘贴保存

---

## ⑤ Fork 仓库（在自己账号下留一份副本）

1. 登录 GitHub 后访问原始项目：`https://github.com/Danielwangyy/AI_Homework_P2P_PPDAI`。  
2. 点击页面右上角 `Fork` → `Create fork`。  
3. 完成后，你将在自己的账号下看到同名仓库，地址如 `https://github.com/你的用户名/AI_Homework_P2P_PPDAI`。

---

## ⑥ 让 Agent 帮你克隆仓库

1. 在自己的仓库页面点击绿色的 `Code` 按钮，复制 `SSH` 或 `HTTPS` 地址。  
2. 回到 Cursor Agent，告诉它：  
   - `请把仓库克隆到 ~/Documents，仓库地址是 <粘贴链接>`  
   （如果你想放在其他文件夹，把 `~/Documents` 换成对应路径。）
3. 克隆完成后，再说：`请进入 ~/Documents/AI_Homework_P2P_PPDAI`。Agent 会帮你切换到项目目录。  
4. 最后说一句：`请告诉我当前所在路径`（即 `pwd`），确认已经进入项目根目录。  
5. 在 Cursor 中选择 `File → Open Workspace...` 打开这个目录，就能在左侧文件树看到所有文件。

---

## ⑦ 创建自己的工作分支

每次要修改项目之前，先让 Agent 新建一个分支，避免直接动主分支：

```
请帮我创建并切换到新分支 feature/my-change
```

你可以把 `feature/my-change` 换成更贴切的名字，比如 `report-update`、`add-eda` 等。Agent 会自动执行 `git checkout -b ...` 并展示结果。

---

## ⑧ 编辑内容

1. 在 Cursor 左侧文件树中找到需要修改的文件，双击打开。  
2. 根据需求直接编辑文字或代码。  
3. 编辑完成后按 `Ctrl/Cmd + S` 保存。  
4. 如果要新建文件，可以在文件树中右键目录 → `New File`，或直接让 Agent 创建：`请帮我在 docs/ 下创建文件 example.md 并写入内容 ...`。

---

## ⑨ 让 Agent 查看、提交你的修改

完成编辑后，依次告诉 Agent：

1. `请告诉我当前的改动`（等同于 `git status`）。  
2. 如果确认要提交：`请把所有改动加入暂存区`（等同于 `git add .`）。  
3. `请用说明 “Update README” 创建一次提交`（Agent 会执行 `git commit -m "Update README"`，文字可以自行调整）。  
4. `请把当前分支推送到我的 GitHub 仓库`（Agent 会执行 `git push origin <分支名>`）。如遇到首次推送提示，按 Agent 给出的指导回应即可。

---

## ⑩ 在 GitHub 发起合并请求（Pull Request）

1. 打开你自己的仓库主页，会看到提示“最近推送了某分支”。点击 `Compare & pull request`。  
2. 确认目标仓库是 `Danielwangyy/AI_Homework_P2P_PPDAI`，目标分支是 `main`。  
3. 填写标题和描述，说明本次修改的目的。  
4. 点击 `Create pull request`。  
5. 等待项目维护者审核。若有修改意见，按照提示更新后，再次让 Agent 帮你提交并推送即可。PR 会自动显示最新改动。

---

## ⑪ 同步主仓库的最新进展

在开始新的工作前，先确保你本地的 `main` 分支是最新的：

1. 仅第一次需要：`请把原作者的仓库加为上游，地址是 git@github.com:Danielwangyy/AI_Homework_P2P_PPDAI.git`（如果走 HTTPS，把地址换成对应链接）。  
2. 每次同步时依次对 Agent 说：  
   - `请获取上游仓库的最新代码`  
   - `请切换到 main 分支`  
   - `请把上游的 main 合并进来`  
   - `请把更新后的 main 推送到我的仓库`  
3. 如果 Agent 提示出现冲突，打开冲突文件手动选择保留的内容，保存后再说：`请继续完成合并并提交`。

---

## ⑫ 常见问题速查

- **Agent 说没找到 git？**  
  按第③步重新安装或配置 Git，然后再让 Agent 执行命令。  
- **推送时要求输入用户名/密码？**  
  GitHub 不再接受账户密码。进入 GitHub → `Settings` → `Developer settings` → `Personal access tokens` 生成 PAT。推送时，用户名填 GitHub 账号，密码位置粘贴 PAT。  
- **Agent 提示权限不足或连接失败？**  
  检查是否已经登录 GitHub（`gh auth status`）、SSH key 是否配置正确、仓库地址是否写错。都可以直接让 Agent 运行相关命令帮你确认。  
- **想讨论或报告问题？**  
  进入仓库页面 → `Issues` → `New issue`，用自然语言描述即可。  

---

## ⑬ 进一步学习资料

- Git 新手教程（中文译版）：<https://git-scm.com/book/zh/v2>  
- GitHub 官方帮助中心：<https://docs.github.com/>  
- Cursor 使用文档：<https://cursor.sh/docs>  

随时在团队群或 PR 评论区提问，大家会一起帮你解决。祝你在机器学习项目中玩的开心、收获满满！


