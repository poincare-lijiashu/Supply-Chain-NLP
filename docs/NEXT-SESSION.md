# NEXT-SESSION 交接快照（2026-09-13 · 难度基准轮后）

## 项目一句话

物流寄递场景寄件文本智能分类系统（简历求职项目）：TF-IDF+RF 基线 → fastText → BERT 微调 →
量化/蒸馏（BiLSTM 学生）→ FastAPI 部署，10 类目 + 置信度阈值转人工复核。

## 环境与运行（v1.2.0）

- conda env：`D:\software_work\conda_envs\SupplyChainNLP`（Python 3.11.16，torch 2.11.0+cu128，RTX 5070 可用）
- **入口已全部迁移为 `python -m` 模块方式（项目根目录执行）**：
  - 数据：`python -m data.generate_data`（seed=2026 确定性复现 18w/1w/1w）
  - 冒烟：`python -m scripts.run_smoke`（隔离 data/smoke + models/smoke，不碰正式权重）→ 6/6
  - 客户端：`python -m src.serving.client`
  - pytest：`python -m pytest -q`（24 passed：test_core 13 + test_smoke 4 + test_security 7）
- 服务：`uvicorn src.serving.app:app --host 127.0.0.1 --port 8004`
- 包结构：`pyproject.toml`（可 `pip install -e .`），全部 `sys.path` 样板已删，根 `conftest.py` 供 pytest

## 2026-09 安全加固轮（评测驱动，全部完成）

1. `torch.load` 9 处显式 `weights_only=True`；int8 产物改存 `state_dict`（修复评测遗漏的保存/加载不匹配 bug）
2. `/predict` 鉴权（`API_AUTH_KEY` 环境变量 + X-API-Key 恒时比较）+ 限流（60/min/IP 滑动窗口）+ 入参上限（≤100 条/单条 ≤512 字）
3. `serve_host` 默认 `127.0.0.1`；Dockerfile 非 root（uid 10001）；Nginx 安全头 + `server_tokens off`
4. 新增 `tests/test_core.py`（蒸馏损失/复核判定/入参校验/前向形状）
5. README 第 2 节数据口径声明（合成演示数据 vs 生产口径）；`docs/DEPLOY.md` 部署指南；fix_log 第二节加固留档；LICENSE(MIT)
6. 详见 `docs/fix_log.md` 第二节 + `docs/release_checklist.md`

## 当前状态（难度基准轮 · 最终）

- **难度基准 v1.2 数据**（generate_data.py）：错字 12%×2处、双物品混填 15%（申报第一项定类目，词序敏感）、
  容器材质属性线索、heldout 漂移、歧义 3-4% —— 目的：让"训练的价值"在阶梯图上可见
- **最终实测阶梯**：RF 87.52 / fastText default 89.11-89.71 / **BERT 96.37** / **蒸馏学生 96.46（反超教师）**
  / int8 96.40 / 剪枝 96.38；fastText autotune 96.4（需 300s 逐场景调参）；延迟 20.5/12.9/1.30ms
- 图表 docs/img/* 已更新（阶梯拉开 + 压缩收益）；README 第 3/4 节按"两条价值曲线"叙事重构
- pytest 24 passed；公开历史已**压平重导出**：origin/main = 041596c 单 init 提交，v1.2.0 标签重指
- 分支模型：main=完整历史；publish=孤儿单提交（发布导出）；推送走本机 7897 代理

## 已知事项 / Backlog

1. `experiments/smoke_run.log` 退出码 0xC0000409（Windows torch/tqdm 关闭阶段原生崩溃）——产物完整，良性（run_smoke 已用 os._exit 规避）
2. LLM 对照实验需复制 `.env.example` → `.env` 填真实 DeepSeek key
3. fasttext 0.9.x 与 numpy 2.x 不兼容补丁在 env 的 FastText.py（重建环境需重打，Dockerfile/CI 已内置）
4. `.env.example` 新增 `API_AUTH_KEY` 占位（服务鉴权开关）

## 重启开发方法

```powershell
cd "D:\workspace_AI\workspace_traecode\xiangmu_traecode\Supply Chain NLP"
D:\software_work\conda_envs\SupplyChainNLP\python.exe -m pytest -q
```

## 分支模型（重要）

- `main`（本地）：完整开发历史 + 全部内容（含内部文档），日常工作分支
- `publish`（本地+远端）：孤儿分支单提交导出树，推送目标 `origin/main`；**不含** NEXT-SESSION / rewrite_matrix / release_checklist
- 后续改动流程：main 上开发 → 需要公开时把变更同步到 publish（cherry-pick 或 read-tree）→ push publish:main
- 网络注意：github.com 直连间歇被断，本机 7897 端口有代理，用 `git -c http.proxy=http://127.0.0.1:7897 push ...`
