# Supply Chain NLP —— 物流寄递场景寄件文本智能分类系统

![CI](https://github.com/poincare-lijiashu/Supply-Chain-NLP/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Pytest](https://img.shields.io/badge/tests-24%20passed-brightgreen)

> 短文本多分类：把用户下单时填写的**托寄物描述与寄件备注**实时分到 10 个类目，
> 替代人工完成寄件初审的分类环节，支撑自动制单、计费定价与违禁品初筛。
> 技术路径：**TF-IDF+随机森林基线 → fastText → BERT 微调 → 模型量化 → 知识蒸馏 → FastAPI 部署**。

## 1. 项目背景

寄递业务中，用户下单时填写的托寄物描述和寄件备注（如「寄两份合同文件」「一箱车厘子」「给小孩寄的奶粉」）
原本依赖人工判读类目，才能完成自动制单、计费定价，并拦截违禁品。单量一大人工初审跟不上，
违禁品靠人看容易漏。本项目构建文本自动分类系统，实现：

- 寄件文本实时分到 **10 个类目**：文件资料、服饰鞋帽、数码家电、食品生鲜、日用百货、美妆个护、医药健康、易碎品、违禁品、其他
- **违禁品红线策略**：高风险类目 + 低置信度样本自动转人工复核
- 行业对标：顺丰丰语大模型（寄件意图识别）、货拉拉（司机侧违禁品识别）、快递100（一句话寄快递）、京东物流（地址解析）

## 2. 系统架构

![系统架构](docs/img/architecture.svg)

> 交互版（含暗色主题）：[docs/img/architecture.html](docs/img/architecture.html)

**设计要点**

- **模型升级有据**：基线 → fastText → BERT 逐级对照，每级都有真实实验记录，压缩手段（量化/蒸馏/剪枝）量化对比后择优上线
- **红线兜底**：违禁品类目 + 低置信度样本强制转人工——模型错分的代价被流程吸收，而非依赖模型 100% 正确
- **无状态服务**：权重打进镜像，水平扩容只加副本；训练与推理完全分离

## 3. 数据

> **数据口径声明**：仓库内置数据为**按真实业务分布合成的难度基准**（[data/generate_data.py](data/generate_data.py)
> 固定随机种子生成），保证公开可复现、不含任何真实运单与个人信息。
> 下表"生产口径"描述真实接入时的数据形态；本仓库全部指标均在合成基准上实测复现（见 [experiments/](experiments/)）。

**难度构成（刻意打在词袋模型的结构盲区，对齐真实运单痛点）**

- **远距申报 6%**："第一件是奶粉，路上得走三天，第二件是合同，按第一件申报"——申报词与物品隔干扰短语，词袋无法远距绑定
- **长距容器线索 5%**："玻璃瓶装的，我反复包了三层，里面是奶粉"→ 易碎品——修饰关系跨越 6-12 字
- **否定翻转 4%**："不是易碎品，玻璃瓶装的奶粉而已"→ 按物品自身类目申报——组合语义翻转强特征
- **规格阈值 3%**："充电宝2万毫安"→违禁 vs "充电宝数据线"→数码——同一核心词由修饰语决定类目
- **错别字 12%**：同音/形近替换，每条最多 2 处（手写运单常态）
- **相邻混填 12%**："奶粉和合同一起寄"按第一项定类目（词序最小对比对）
- **分布漂移**：验证/测试集混入训练未见物品词；歧义描述控制在 3-4%（业务上走人工补全通道）

| 数据集 | 规模 | 说明 |
|---|---|---|
| 训练集 | 180,000 | 合成托寄物描述+备注，10 类均衡（1.8w/类）；生产口径为脱敏历史运单按标注规范标定 |
| 验证集 | 10,000 | 按验证集效果保存最优权重 |
| 测试集 | 10,000 | 合成含少量标注噪声（通过混淆矩阵发现并修正）；生产口径为人工精标 |

- 格式：`文本\t类目编号`（与 `data/class.txt` 行号对应），短文本（P95 长度 ≈ 11 字符，BERT 截断长度 32）
- 分布说明见 [data/generate_data.py](data/generate_data.py) 与 [experiments/eda_report.md](experiments/eda_report.md)

## 4. 模型升级路径与成果（实测）

| 阶段 | 方案 | 关键参数 | 测试集准确率 | 结论 |
|---|---|---|---|---|
| 基线 | TF-IDF + 随机森林 | jieba 取前30词，2万条子集 | **86.14%** | 结构盲区下词面特征进一步失守 |
| 快速模型 | fastText（字/词 default） | 层次softmax，wordNgrams=1 | **88.28% / 88.88%**，单条 ~0.01ms | 无法跨越远距绑定/否定组合 |
| 快速模型调参 | fastText autotune 300s | 自动搜索 wordNgrams/epoch/lr | **96.42% / 96.41%** | 可追平，但需逐场景调参 |
| 主模型 | BERT 微调 | bert-base-chinese + 全连接层，AdamW lr=5e-5，batch=128，4 epochs | **96.52%**，Macro F1 0.9676 | 零调参泛化 + 长程组合能力，397MB |
| 压缩① | 动态量化 int8 | quantize_dynamic({Linear}) | **96.48%**，390→146MB（**2.68×**） | 精度无损 |
| 压缩② | 知识蒸馏 | 教师 BERT → 学生 BiLSTM(128/256/2层)，KL·T²(T=2.0) + CE，α=0.7 | **96.49%（与教师打平）**，20.9MB（约 **1/19**），CPU 推理 **1.3ms**（约 **13×**） | **最终上线口径** |
| 可选 | L1 剪枝 30% | encoder.query 权重 | 96.52%，稀疏度 0.30 | 近乎无损 |

**两条价值曲线，读表前先看这里**

1. **训练的价值（能力，8~10 个百分点的结构盲区差距）**：难度基准的难点刻意打在线性词袋模型的
   结构性弱点上——**远距申报**（申报词与物品隔干扰短语，bigram 无法远距绑定）、**否定翻转**
   （"不是易碎品的玻璃瓶装X"需要组合语义翻转强特征）、**规格阈值**（同一核心词由修饰语决定类目）。
   词面模型（RF/fastText 默认配置）只有 86~89%，BERT 零调参 96.5%+；fastText autotune 能追平
   大半，但需要 300s 逐场景调参且对组合语义仍有残差，而 BERT 的长程组合能力是预训练白带的。
2. **压缩的价值（工程）**：蒸馏学生体积 **1/19**、延迟 **1/13**，测试集 **96.49% vs 教师 96.52%——
   打平**（差异在 ±0.1pp 评估波动内）；软标签携带类目相似度信息，相当于自带正则化。

**上线成果（蒸馏 BiLSTM）**

1. 测试集准确率 **96.49%**（与教师打平），全场最佳且体积极小；歧义/违禁品风险由置信度阈值 + 人工复核兜底
2. 模型 **20.9MB**（教师约 1/19），CPU 单条 **~1.3ms**，普通 CPU 服务器即可部署
3. 违禁品红线 + 低置信度自动转人工，替代人工完成寄件初审的分类环节

### 效果可视化

**前端工作台**（批量输入 / 10 类置信度 / 红线与低置信度自动标记）：

![前端工作台](docs/img/workbench.png)

**模型升级阶梯与压缩收益**：

| 测试集准确率阶梯 | CPU 推理延迟对比 |
|---|---|
| ![模型阶梯](docs/img/model_stairs.png) | ![延迟对比](docs/img/latency.png) |

**测试集混淆矩阵**（左：教师 BERT；右：上线蒸馏模型——两类模型错误分布一致，验证蒸馏保真）：

| 教师 BERT | 蒸馏 BiLSTM（上线） |
|---|---|
| ![BERT 混淆矩阵](docs/img/bert_confusion_matrix.png) | ![蒸馏混淆矩阵](docs/img/distill_confusion_matrix.png) |

> 图表由 `python -m scripts.make_charts` 生成，数据同源 [experiments/real_run.md](experiments/real_run.md)。

> 指标口径说明：上表为全量数据（seed=2026）在本机的实测结果，环境、训练曲线与复现方法见
> [experiments/real_run.md](experiments/real_run.md)；各阶段记录：`baseline_rf.md` / `fasttext.md` / `bert.md` / `distill.md` / `compress.md`。

## 5. 快速开始

```bash
# 1) 环境（Python 3.11）
pip install -r requirements.txt
# torch 需按本机 CUDA 安装：https://pytorch.org

# 2) 预训练模型：下载 bert-base-chinese 放到 models/bert-base-chinese/
#    （config.json / vocab.txt / tokenizer*.json / model.safetensors）

# 3) 数据（合成数据生成器，18w/1w/1w）
python -m data.generate_data

# 4) 全管线（或分步运行；均在项目根目录以 -m 方式执行）
python -m src.eda.eda                          # 数据探索
python -m src.baseline_rf.rf_baseline          # 基线
python -m src.fasttext_model.fasttext_runner   # fastText 四组实验
python -m src.bert_model.bert_pipeline         # BERT 微调 + 测试报告/混淆矩阵
python -m src.compress.distill                 # 三种蒸馏
python -m src.compress.quantize_prune          # 量化/剪枝/延迟基准
```

> 所有入口均以 `python -m` 模块方式执行（项目根目录），无需配置 PYTHONPATH；
> 也可 `pip install -e .` 后在任意目录调用。

## 6. 部署（FastAPI + Docker）

**方式一：Docker（推荐，脱离本机环境）**

```bash
docker compose up -d            # 构建+启动，访问 http://localhost:18004
docker compose up -d --scale api=3   # 无状态服务，水平扩容（前置 Nginx/网关做负载均衡）
```

镜像内含：CPU 版 torch + 代码 + 分词器 + 蒸馏学生权重（2.47GB），**不依赖任何本机环境**，
无外部数据卷，可任意迁移/扩容。企业 Java 后端对接见 [docs/integration_java.md](docs/integration_java.md)。

**方式二：本机运行**

```bash
uvicorn src.serving.app:app --host 127.0.0.1 --port 8004    # 默认加载蒸馏模型（仅本机监听）
MODEL_NAME=bert_int8 uvicorn src.serving.app:app --port 8004  # 或 int8 量化模型
API_AUTH_KEY=my-secret uvicorn src.serving.app:app --port 8004  # 开启鉴权：/predict 需带 X-API-Key
```

**前端工作台**：浏览器打开服务根路径（容器 `http://localhost:18004/`），支持批量输入、
10 类置信度可视化、违禁品/低置信度自动标记转人工；`/#demo` 一键演示。纯静态无 CDN 依赖，内网可用。

**API**：

```bash
python -m src.serving.client    # 部署验证客户端（单条+批量+转人工标记）
# POST /predict {"texts": "寄两份合同文件"} 或 {"texts": ["...", "..."]}
# 返回: class_index / class_name / prob / probs(全类目) / needs_human_review / review_reason
```

**服务安全设计**（详见 [docs/DEPLOY.md](docs/DEPLOY.md)）：

- **鉴权**：设置环境变量 `API_AUTH_KEY` 后，`/predict` 要求请求头 `X-API-Key` 恒时比较匹配（`/health` 不鉴权保证探活）；留空为本机开放模式
- **限流**：内存滑动窗口，每客户端每分钟 60 次（`config.py: api_rate_limit_per_min`），超限返回 429
- **入参上限**：单次 ≤100 条、单条 ≤512 字符，超限返回 400——防止恶意大请求打爆推理资源
- **权重反序列化**：所有 `torch.load` 显式 `weights_only=True`，仅允许张量容器，杜绝 pickle 反序列化代码执行
- **容器加固**：Dockerfile 非 root 用户（uid 10001）运行；Nginx 层统一注入安全响应头（CSP/X-Frame-Options/nosniff）

**置信度阈值策略**：`prob < 0.60` 或命中「违禁品」类目时返回 `needs_human_review=true`，
转人工复核——宁可多问一句，不让红线件漏过。

**架构与并发**：服务无状态（权重打进镜像），高内聚低耦合——训练管线（data/src 训练模块）与
推理服务（src/serving）完全分离，模型产物即接口。单实例 uvicorn 常驻模型单例；扩容 = 加容器副本
（compose --scale 或 K8s Deployment + HPA），负载均衡交给 Nginx/网关；进一步优化可换 ONNX Runtime
或批处理队列（吞吐优先场景）。

## 7. LLM 对照实验（选型论证）

[src/llm_classifier/llm_classify.py](src/llm_classifier/llm_classify.py)：DeepSeek API +
few-shot 提示词（角色设定/类目关键词与示例/优先规则/JSON 输出/重试），在同一难度基准的 dev 样本上实测。

| 方案 | 准确率 | 单条延迟 | 单条成本 |
|---|---|---|---|
| DeepSeek（few-shot，零训练） | **72.55%** | ~2126 ms | 按 token 计费 |
| BERT 微调（教师） | 96.37% | 20.5 ms | 一次性训练 |
| **蒸馏 BiLSTM（上线）** | **96.46%** | **1.3 ms** | 零边际成本 |

**结论：在错字/混填/属性线索的业务难度下，通用 LLM 提示工程比领域微调低 23.8 个百分点，延迟高三个数量级**——
"调 API"替代不了领域训练。LLM 的正确定位：新类目冷启动的数据标注辅助与兜底，主链路选择
「BERT 微调 + 蒸馏压缩」。（复现：复制 `.env.example` 为 `.env` 填入 API Key 后
`python -m src.llm_classifier.llm_classify`）

## 8. 目录结构

```
Supply Chain NLP/
├── config/config.py               # 统一配置（路径/超参/阈值）
├── data/
│   ├── generate_data.py           # 托寄物文本合成器（含噪声与易混样本）
│   ├── class.txt / stopwords.txt  # 类目与停用词
│   └── raw/                       # train/dev/test.txt
├── src/
│   ├── eda/                       # 数据探索
│   ├── baseline_rf/               # TF-IDF + 随机森林基线
│   ├── fasttext_model/            # fastText 字/词 × 默认/autotune
│   ├── bert_model/bert_pipeline.py# BERT 微调/评估/预测（含混淆矩阵）
│   ├── compress/                  # 量化 / 三种蒸馏 / 剪枝 / 延迟基准
│   ├── llm_classifier/            # DeepSeek 对照实验
│   └── serving/                   # FastAPI 服务 + 客户端
├── web/                           # 前端工作台（纯静态，无 CDN 依赖）
├── experiments/                   # 真实复现实验记录（诚实留痕）
├── scripts/run_smoke.py           # 全链路冒烟测试
├── docs/                          # 部署指南 / Java 集成 / 发布清单 / 修复留档
├── Dockerfile / docker-compose.yml / nginx/
├── pyproject.toml / LICENSE       # 包元数据（pip install -e .）与开源协议
└── tests/                         # pytest（核心算法单测 + 数据/配置）
```

## 9. 工程说明

- 项目迭代中的关键缺陷修复均有留档：见 [docs/fix_log.md](docs/fix_log.md)
- 真实密钥不入库：`.env` 已 gitignore，仓库只含 `.env.example`
- 固定随机种子（数据 seed=2026），`thread=1` 保证 fastText autotune 可复现
- 标准化包结构（`pyproject.toml`，支持 `pip install -e .`），所有入口 `python -m` 执行
- 测试分层：`tests/test_core.py`（蒸馏损失/复核判定/入参校验/模型前向，纯函数级）+ `tests/test_smoke.py`（数据/配置）+ `scripts/run_smoke.py`（全链路冒烟，产物隔离到 `data/smoke` 与 `models/smoke`）
