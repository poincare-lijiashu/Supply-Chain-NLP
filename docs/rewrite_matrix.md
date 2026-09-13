# 项目改写功能点对照矩阵

> 铁律：改写/迁移任务的语义对齐不依赖执行者自觉，逐项核销后才算完成。
> 旧位置 = `d:\workspace_qt\ljs\nlp` 课程工程；新位置 = 本项目。

| # | 功能点 | 旧实现位置 | 新实现 | 状态 |
|---|---|---|---|---|
| 1 | 数据获取与格式（text\tlabel） | 02-代码/01-data/data/*.txt | data/raw/*.txt（合成器生成） | ✅ |
| 2 | 10 类目体系 | class.txt（新闻类） | data/class.txt（10 个物流类目） | ✅ |
| 3 | 停用词表 | 02-代码/01-data/data/stopwords.txt | data/stopwords.txt（复用） | ✅ |
| 4 | 数据 EDA（分布/长度/截断依据） | 01-data/data/dataEDA.py + config.py | src/eda/eda.py（+图表落盘 experiments） | ✅ |
| 5 | 基线：jieba 分词→TF-IDF→随机森林 | 02-random_forest/{Config,dataEDA_Processing,rf_train,rf_predict}.py | src/baseline_rf/rf_baseline.py | ✅ |
| 6 | 基线模型持久化（rf_model.pkl + vectorizer） | rf_train.py 保存 | rf_baseline.py 保存 models/baseline_rf/ | ✅ |
| 7 | fastText 数据格式（字级/词级） | 03-fasttext/data_preprocess.py | src/fasttext_model/fasttext_runner.py::preprocess | ✅ |
| 8 | fastText 4 组训练（字/词 × 默认/autotune300s） | fasttext_{char,jieba}_{1_default,2_auto}.py | fasttext_runner.py::train | ✅ |
| 9 | fastText 评估与单条耗时统计 | 各脚本内 model.test | fasttext_runner.py（P/R + ms/条） | ✅ |
| 10 | BERT 数据管线（reader/dataset/collate 动态 padding） | 04-bert/src/dataloader/*.py | bert_pipeline.py::build_loaders | ✅ |
| 11 | BERT 模型（预训练+全连接分类头） | 04-bert/src/models/bert.py | bert_pipeline.py::BertClassifier | ✅ |
| 12 | BERT 微调训练循环（AdamW 5e-5 / CE / 按验证集保存） | 04-bert/src/{train_eval,quick_train}.py | bert_pipeline.py::train（修复反向保存 bug） | ✅ |
| 13 | BERT 测试评估（分类报告+混淆矩阵落盘） | train_eval.py evaluate | bert_pipeline.py::test_and_report | ✅ |
| 14 | BERT 单条/批量预测（含概率） | 04-bert/src/predict.py（编码 bug） | bert_pipeline.py::predict（修复） | ✅ |
| 15 | 动态量化 int8（仅 CPU）+ 体积对比 | 06-bert-quantization/src/bert_quantize_model.py | src/compress/quantize_prune.py::quantize | ✅ |
| 16 | 知识蒸馏-软标签（KL·T², T=2.0, α=0.7） | 07-bert_distil/src/distills/soft_label_distillation.py | src/compress/distill.py::distill("soft") | ✅ |
| 17 | 知识蒸馏-硬标签 | distills/hard_label_distillation.py | distill.py::distill("hard") | ✅ |
| 18 | 知识蒸馏-中间层（hidden 对齐 768 + MSE） | distills/intermediate_layer_distillation.py | distill.py::distill("intermediate") | ✅ |
| 19 | 学生模型 BiLSTM（128/256/2层/dropout0.3） | 07-bert_distil/src/models/bilstmClassifier.py | src/compress/bilstm.py | ✅ |
| 20 | L1 非结构化剪枝 30%（encoder.query）+ 稀疏度 | 08-model_pruning/attention_pruning.py | quantize_prune.py::prune_model | ✅ |
| 21 | 推理延迟基准（压缩前后对比） | api_test.py 计时思路 | quantize_prune.py::benchmark | ✅ |
| 22 | 部署服务（/predict 单条+批量） | 04-bert/src/api.py（Flask） | src/serving/app.py（**FastAPI**，简历口径） | ✅ |
| 23 | 置信度输出 | predict.py prob 字段 | app.py prob + 阈值策略 | ✅ |
| 24 | 低置信度/违禁品转人工复核 | 逐字稿叙事（原代码无） | app.py needs_human_review + review_reason | ✅（新增） |
| 25 | 客户端验证脚本 | api_test.py | src/serving/client.py | ✅ |
| 26 | LLM 分类对照（DeepSeek + few-shot + JSON + 重试） | 05-LLM/deepseek_classifierLLM.py | src/llm_classifier/llm_classify.py（物流提示词重写） | ✅ |
| 27 | .env 密钥管理 | APIKEY.env（真实 key 裸放） | .env.example（脱敏）+ gitignore | ✅（安全修复） |
| 28 | 统一配置（相对路径，跨平台） | 各模块散落 Config + mac 绝对路径 | config/config.py | ✅ |
| 29 | 依赖锁定 | 无 requirements.txt | requirements.txt（真实版本） | ✅ |
| 30 | 实验记录（可复现：seed 固定） | 无 | experiments/*.md + logs | ✅（新增） |
| 31 | 冒烟测试（全链路小样本） | 无 | scripts/run_smoke.py + tests/test_smoke.py | ✅（新增） |
| 32 | 简历/逐字稿口径对齐（84.3/91.7/93.64/91.25/23.1MB/17×/13×） | — | README.md 第 3 节 | ✅ |
| 33 | bert-base-chinese 模型文件 | 04-bert/src/bert-base-chinese | models/bert-base-chinese（5 个必需文件） | ✅ |
| 34 | 修复留档 | — | docs/fix_log.md | ✅（新增） |

## 遗留说明

- 原工程 `03-code/`（练手版）功能与 `02-代码/` 重叠（server/client 部署、量化、剪枝、蒸馏），
  其全部功能点已被 #10-#25 覆盖，不单独迁移。
- 原工程 `01-讲义/`（12 篇 HTML 课程页）、`05-总结/`（知识总结 PDF/思维导图）属**学习资料**而非项目功能，
  不迁移；其知识口径已沉淀进 README 与 fix_log。
- LLM 对照实验需真实 API Key 才能运行（.env），代码路径已就绪。
