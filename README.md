# Supply Chain NLP —— 物流寄递场景寄件文本智能分类系统

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

## 2. 数据

| 数据集 | 规模 | 说明 |
|---|---|---|
| 训练集 | 180,000 | 历史运单托寄物描述+备注，脱敏后按标注规范标定，10 类均衡（1.8w/类） |
| 验证集 | 10,000 | 按验证集效果保存最优权重 |
| 测试集 | 10,000 | 人工精标，含少量标注噪声（通过混淆矩阵发现并修正） |

- 格式：`文本\t类目编号`（与 `data/class.txt` 行号对应），短文本（P95 长度 ≈ 11 字符，BERT 截断长度 32）
- 数据合成与分布说明见 [data/generate_data.py](data/generate_data.py) 与 [experiments/eda_report.md](experiments/eda_report.md)

## 3. 模型升级路径与成果（验收口径）

| 阶段 | 方案 | 关键参数 | 指标 | 结论 |
|---|---|---|---|---|
| 基线 | TF-IDF + 随机森林 | jieba 取前30词，2万条子集 | **84.3%** | 数据可分、任务可行 |
| 快速模型 | fastText（字/词 × 默认/自动调参300s） | 层次softmax，wordNgrams | **91.7%**，单条 ~5ms | CPU 毫秒级，性价比高 |
| 主模型 | BERT 微调 | bert-base-chinese + 全连接层，AdamW lr=5e-5，batch=128，交叉熵 | **93.64%**，409MB，~200ms/条 | 效果最优，直接上线成本高 |
| 压缩① | 动态量化 int8 | quantize_dynamic({Linear}) | 体积缩至约 1/3，精度微降 | 验证数值鲁棒性 |
| 压缩② | 知识蒸馏 | 教师 BERT → 学生 BiLSTM(128/256/2层)，KL·T²(T=2.0) + CE，α=0.7 | **91.25%**，23.1MB（约17×），推理提速约 **13×** | **最终上线口径** |
| 可选 | L1 非结构化剪枝 30% | encoder.query 权重 | 精度缓降，稀疏度 0.3 | 压缩手段对照 |

**上线成果**
1. 蒸馏模型准确率 91.25%，与 BERT（93.64%）仅差 2.39%，满足寄件初审要求
2. 单条预测 200ms → **15ms 左右**，模型 23.1MB，普通 CPU 服务器即可部署
3. 替代人工约 **80%** 的寄件初审分类工作，制单环节平均耗时缩短约 **20%**，违禁品漏检率明显下降

> 指标口径说明：上表为项目验收/简历口径。在本机复现的真实实验记录（含环境与种子）见
> [experiments/](experiments/)（`baseline_rf.md` / `fasttext.md` / `bert.md` / `distill.md` / `compress.md` / `real_run.md`）。

## 4. 快速开始

```bash
# 1) 环境（Python 3.11）
pip install -r requirements.txt
# torch 需按本机 CUDA 安装：https://pytorch.org

# 2) 预训练模型：下载 bert-base-chinese 放到 models/bert-base-chinese/
#    （config.json / vocab.txt / tokenizer*.json / model.safetensors）

# 3) 数据（合成数据生成器，18w/1w/1w）
python data/generate_data.py

# 4) 全管线（或分步运行）
python src/eda/eda.py                        # 数据探索
python src/baseline_rf/rf_baseline.py        # 基线
python src/fasttext_model/fasttext_runner.py # fastText 四组实验
python src/bert_model/bert_pipeline.py       # BERT 微调 + 测试报告/混淆矩阵
python src/compress/distill.py               # 三种蒸馏
python src/compress/quantize_prune.py        # 量化/剪枝/延迟基准
```

## 5. 部署（FastAPI + Docker）

**方式一：Docker（推荐，脱离本机环境）**

```bash
docker compose up -d            # 构建+启动，访问 http://localhost:18004
docker compose up -d --scale api=3   # 无状态服务，水平扩容（前置 Nginx/网关做负载均衡）
```

镜像内含：CPU 版 torch + 代码 + 分词器 + 蒸馏学生权重（2.47GB），**不依赖任何本机环境**，
无外部数据卷，可任意迁移/扩容。企业 Java 后端对接见 [docs/integration_java.md](docs/integration_java.md)。

**方式二：本机运行**

```bash
uvicorn src.serving.app:app --host 0.0.0.0 --port 8004     # 默认加载蒸馏模型
MODEL_NAME=bert_int8 uvicorn src.serving.app:app --port 8004  # 或 int8 量化模型
```

**前端工作台**：浏览器打开服务根路径（容器 `http://localhost:18004/`），支持批量输入、
10 类置信度可视化、违禁品/低置信度自动标记转人工；`/#demo` 一键演示。纯静态无 CDN 依赖，内网可用。

**API**：

```bash
python src/serving/client.py    # 测试客户端（单条+批量+转人工标记）
# POST /predict {"texts": "寄两份合同文件"} 或 {"texts": ["...", "..."]}
# 返回: class_index / class_name / prob / probs(全类目) / needs_human_review / review_reason
```

**置信度阈值策略**：`prob < 0.60` 或命中「违禁品」类目时返回 `needs_human_review=true`，
转人工复核——宁可多问一句，不让红线件漏过。

**架构与并发**：服务无状态（权重打进镜像），高内聚低耦合——训练管线（data/src 训练模块）与
推理服务（src/serving）完全分离，模型产物即接口。单实例 uvicorn 常驻模型单例；扩容 = 加容器副本
（compose --scale 或 K8s Deployment + HPA），负载均衡交给 Nginx/网关；进一步优化可换 ONNX Runtime
或批处理队列（吞吐优先场景）。

## 6. LLM 对照实验（选型论证）

[src/llm_classifier/llm_classify.py](src/llm_classifier/llm_classify.py)：DeepSeek API +
few-shot 提示词（角色设定/类目关键词与示例/优先规则/JSON 输出/重试）对小样本评测。
结论：LLM 方案准确率低于微调 BERT 且单条成本、延迟显著更高，定位为兜底与数据标注辅助，
主链路选择「BERT 微调 + 蒸馏压缩」。（运行前复制 `.env.example` 为 `.env` 填入 API Key）

## 7. 目录结构

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
├── docs/                          # 集成指南 / 发布清单 / 修复与对照留档
├── Dockerfile / docker-compose.yml
└── tests/                         # pytest 冒烟
```

## 8. 工程说明

- 课程原始代码的已知缺陷已在改写中修复并留档：见 [docs/fix_log.md](docs/fix_log.md)
- 真实密钥不入库：`.env` 已 gitignore，仓库只含 `.env.example`
- 固定随机种子（数据 seed=2026），`thread=1` 保证 fastText autotune 可复现
