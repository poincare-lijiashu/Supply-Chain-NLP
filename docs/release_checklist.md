# GitHub 发布清单（去 demo 化 + 去隐私化）

## 隐私核查（发布前逐项过）

- [x] **git 作者身份**：提交时用 `git -c user.name="..." -c user.email="<noreply>"` 覆盖，
      禁止全局配置的手机号邮箱 `19231255485@163.com` 进入提交对象
- [x] 真实 API Key：仓库内只有 `.env.example`（占位符），`.env` 已 gitignore；原课程工程
      `APIKEY.env` 未复制进本项目
- [x] 个人信息：README/文档无姓名、电话、公司真名（简历叙事中公司为"北京xxxx 公司"占位）
- [x] 数据：`data/raw/`（合成数据）不入库（体积原因）；合成器可一键重建
- [x] 模型权重：`models/` 不入库（bert-base-chinese 392MB + checkpoints），README 给下载说明
- [x] 仓库元数据：描述不含个人信息；不关联真实公司/学校

## 去 demo 化

- [x] 一键运行路径完整：数据生成 → 训练 → 评估 → 部署，每步有真实实验记录（experiments/）
- [x] 双层口径诚实：README 第 2 节数据口径声明（合成演示数据 vs 生产口径）+ experiments/real_run.md 本机复现，不冒充生产系统
- [x] 工程件齐全：requirements 锁版本、pyproject.toml（pip install -e .）、Dockerfile/compose、FastAPI/OpenAPI、pytest、冒烟脚本
- [x] 修复留档 docs/fix_log.md（对课程代码的问题如实标注来源与修法；2026-09 安全加固轮见同文件第二节）
- [x] LICENSE（MIT，2026-09-13 确认）
- [x] 服务安全加固（2026-09）：API_AUTH_KEY 鉴权 + 限流 + 入参上限 + torch.load weights_only + Docker 非 root + Nginx 安全头（详见 docs/DEPLOY.md 第 4 节）
- [ ] CI：`.github/workflows/ci.yml` 已提供（pytest + 冒烟）；**首跑绿了才挂徽章**——没有绿 CI 不挂 CI 徽章
- [ ] Release：tag 用 v 前缀语义化版本（如 v1.2.0），notes 分「核心功能/质量与安全/复现口径」

## 仓库元数据（gh token 权限不足，需在 GitHub Settings 手动填写）

- **About 描述**：物流寄件文本智能分类系统：BERT 微调 → 知识蒸馏（96.37%，20.9MB，CPU 1.3ms）→ FastAPI + Docker 部署，违禁品红线自动转人工
- **Topics**：`nlp` `bert` `knowledge-distillation` `text-classification` `fastapi` `logistics` `pytorch` `docker`
- **Website**（可选）：无

## 发布排除清单（孤儿分支导出时不入库）

- [ ] `docs/NEXT-SESSION.md`（内部交接快照）
- [ ] `docs/rewrite_matrix.md`（内部功能对照矩阵，含本地路径）
- [ ] `docs/release_checklist.md`（本清单）
- [ ] `.env`、`data/`（raw/processed/smoke）、`models/`（权重与缓存，README 有重建说明）

## 发布流程（孤儿分支单提交导出，防历史泄漏）

```powershell
git checkout --orphan publish
git add -A
# 双重核对 staged 清单：数量比对 + 可疑模式 grep（.env/密钥/数据/权重/个人信息/demo化字样）
git status --short
git -c user.name="<GitHub用户名>" -c user.email="<GitHub noreply>@users.noreply.github.com" commit -m "init: supply chain NLP v1.2.0"
git push -u origin publish:main
```

本地 main 若有完整历史，**不要** push 历史分支；公开仓库只保留孤儿单提交。
