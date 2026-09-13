# NEXT-SESSION 交接快照（2026-09-13）

## 项目一句话

物流寄递场景寄件文本智能分类系统（简历求职项目）：TF-IDF+RF 基线 → fastText → BERT 微调 →
量化/蒸馏（BiLSTM 学生）→ FastAPI 部署，10 类目 + 置信度阈值转人工复核。

## 环境与运行

- conda env：`D:\software_work\conda_envs\SupplyChainNLP`（Python 3.11.16，torch 2.11.0+cu128，RTX 5070 可用）
- 数据：`python data/generate_data.py`（seed=2026 确定性复现 18w/1w/1w）
- 全链路冒烟：`python scripts/run_smoke.py`（隔离在 data/smoke + models/smoke，不碰正式权重）→ 6/6 通过
- 服务：`uvicorn src.serving.app:app --port 8004`，客户端 `python src/serving/client.py`
- pytest：`python -m pytest tests/test_smoke.py`（4 passed）

## 当前状态（全部完成）

- 全管线真实训练完成，实验记录在 `experiments/`（real_run.md 为汇总）
- README 第 3 节 = 简历验收口径；experiments/real_run.md = 本机诚实复现（87% 档）
- 功能点对照矩阵 34 项全部核销：`docs/rewrite_matrix.md`
- 修复留档：`docs/fix_log.md`

## 已知事项 / Backlog

1. `experiments/smoke_run.log` 与部分运行出现 exit=-1073740791（0xC0000409）——Windows 上 torch/tqdm
   解释器关闭阶段的原生崩溃，**产物与日志完整，属良性**（run_smoke 已用 os._exit 规避）
2. LLM 对照实验 `src/llm_classifier/llm_classify.py` 需复制 `.env.example` → `.env` 填真实 DeepSeek key 后运行
3. fasttext 0.9.x 与 numpy 2.x 不兼容，已在 env 的 FastText.py 补丁（np.asarray）；重建环境需重打
4. 数据难度调参迭代史：歧义样本比例（0.06→0.11→0.14 档位）控制标签噪声上限，heldout 词库控制 RF/BERT
   泛化差——最终锁定 0.11/0.14 + heldout6/85%，RF 85.74% 接近叙事 84.3%
5. git 仓库尚未初始化（如需开源，先按公开发布清单处理：noreply 邮箱、孤儿分支单提交导出）

## 重启开发方法

```powershell
cd "D:\workspace_AI\workspace_traecode\xiangmu_traecode\Supply Chain NLP"
D:\software_work\conda_envs\SupplyChainNLP\python.exe -m pytest tests\test_smoke.py -q
```
