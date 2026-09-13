# 工程演进与修复留档

项目从初版原型迭代到当前版本的过程中，以下缺陷在早期实现中存在，已在重构中修复并回归验证。

| # | 位置（初版） | 问题 | 修复 |
|---|---|---|---|
| 1 | bert 训练脚本 | `best_dev_f1=10` 且 `if f1score < best_dev_f1` 保存——**保存逻辑反向**（越训越差才保存），且命中即 break 提前终止训练 | 验证集 acc **提升才保存**最优权重，正常遍历全部 epoch |
| 2 | evaluate() | 遍历首个 batch 后 `break`，验证/测试只在 1 个 batch 上进行 | 完整遍历 |
| 3 | bert 训练脚本 | `if (i + 1) % 1 == 0` 每个 batch 都做全量验证+classification_report，拖慢训练 | 每个 epoch 结束验证一次 |
| 4 | predict 脚本 | 循环内 `tokenizer.encode_plus(texts, ...)` 误用**整批 texts**（应为单条 `text`）；`max_sequence=512` 参数名错误 | 批量 tokenize（padding/truncation/max_length） |
| 5 | 量化训练脚本 | `best_dev_f1=0` 配 `if f1score < best_dev_f1` 永不保存；每个 batch `torch.save` 后无条件 `break`；重复调用 optimizer.step | 统一训练循环（同 #1） |
| 6 | 多处 Config | 绝对路径硬编码（`/Users/mac/...`），跨机器不可运行 | 全部相对 PROJECT_ROOT，跨平台 |
| 7 | RF 基线 | TF-IDF 在 2 万子集上 fit_transform 后直接 train_test_split（验证集来自训练子集，非独立 dev 集） | 基线在独立 dev 集上评估 |
| 8 | fasttext api | 模型路径硬编码到本地目录；`res[0][0][9:]` 魔法数字截断 `__label__` | 路径走配置；label 解析用前缀剥离 |
| 9 | 场景口径 | 初版为通用新闻标题分类示例 | 聚焦物流寄件托寄物文本分类，类目/数据/文案全部对齐 |
| 10 | 密钥 | 真实 API Key 文件放在工作区内 | 仅保留 `.env.example`，真实密钥不入库 |

另：fasttext 0.9.x 与 numpy 2.x 不兼容（`np.array(probs, copy=False)`），已在运行环境补丁为
`np.asarray`；requirements 未锁 numpy 降级（torch 2.11 / transformers 5.17 依赖 numpy 2.x）。

---

# 安全与工程加固留档（2026-09 评测驱动）

以下问题来自第三方架构/安全评测（含两条评测遗漏的自查发现），已全部修复。

| # | 位置 | 问题 | 修复 |
|---|---|---|---|
| 1 | serving/app.py 等 9 处 | `torch.load` 未显式 `weights_only`——旧版 torch 反序列化可执行任意代码（pickle 面） | 全部显式 `weights_only=True`（仅允许张量容器） |
| 2 | serving/app.py | `/predict` 无鉴权/限流/入参上限，公网暴露即资源滥用 | `API_AUTH_KEY` + X-API-Key 恒时比较；滑动窗口限流 60/min/IP；单次 ≤100 条、单条 ≤512 字符 |
| 3 | config/config.py | `serve_host` 默认 `0.0.0.0`（本机起服务即对局域网开放） | 默认 `127.0.0.1`；容器 CMD 显式传 `0.0.0.0`（容器内网） |
| 4 | Dockerfile | 容器以 root 运行 | 非 root 用户 `appuser`（uid 10001） |
| 5 | nginx/nginx.conf | 无安全响应头、版本泄露 | CSP('self')/X-Frame-Options/nosniff/Referrer-Policy + `server_tokens off` + X-Real-IP 透传 |
| 6 | compress/quantize_prune.py | **评测遗漏自查发现**：`torch.save(q_model)` 保存整模型 pickle，而 serving 端用 `load_state_dict` 加载——类型不匹配且 `weights_only=True` 下必崩（`MODEL_NAME=bert_int8` 路径潜伏坏损） | 改为 `torch.save(q_model.state_dict())`，与加载方式匹配 |
| 7 | 12 处源码 | `sys.path.insert` 样板重复（包结构不规范） | `pyproject.toml` + `__init__.py` 全覆盖 + 根 `conftest.py`；入口统一 `python -m`；样板全删 |
| 8 | tests/ | 核心算法零单测（只有 config/数据格式 4 例） | 新增 `tests/test_core.py`：蒸馏损失公式（对齐/退化/T² 温度）、复核判定三分支、入参校验、配置边界、学生模型前向形状 |
| 9 | README | 数据口径"历史运单脱敏"叙事与"合成数据"事实存在混读空间 | 第 2 节顶部增加数据口径声明：仓库内为合成演示数据，生产口径单列 |

Git 身份提醒：本仓库历史提交作者邮箱含手机号，公开推送必须走
[release_checklist.md](release_checklist.md) 的孤儿分支单提交导出流程（noreply 身份）。
