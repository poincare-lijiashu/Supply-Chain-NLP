# 改写修复留档（课程原代码 → 本项目）

改写自 NLP 文本分类课程工程，以下问题在原代码中存在、本项目已修复。

| # | 位置（原） | 问题 | 修复 |
|---|---|---|---|
| 1 | 04-bert/src/train_eval.py | `best_dev_f1=10` 且 `if f1score < best_dev_f1` 保存——**保存逻辑反向**（越训越差才保存），且命中即 break 提前终止训练 | 验证集 acc **提升才保存**最优权重，正常遍历全部 epoch |
| 2 | 04-bert/src/train_eval.py evaluate() | 遍历首个 batch 后 `break`，验证/测试只在 1 个 batch 上进行 | 完整遍历 |
| 3 | 04-bert/src/train_eval.py | `if (i + 1) % 1 == 0` 每个 batch 都做全量验证+classification_report，拖慢训练 | 每个 epoch 结束验证一次 |
| 4 | 04-bert/src/predict.py | 循环内 `tokenizer.encode_plus(texts, ...)` 误用**整批 texts**（应为单条 `text`）；`max_sequence=512` 参数名错误 | 批量 tokenize（padding/truncation/max_length） |
| 5 | 06-bert-quantization/src/train.py | `best_dev_f1=0` 配 `if f1score < best_dev_f1` 永不保存；每个 batch `torch.save` 后无条件 `break`；重复调用 optimizer.step | 统一训练循环（同 #1） |
| 6 | 多处 Config | 讲师 macOS 绝对路径硬编码（`/Users/mac/...`） | 全部相对 PROJECT_ROOT，跨平台 |
| 7 | 02-random_forest/rf_train.py | TF-IDF 在 2 万子集上 fit_transform 后直接 train_test_split（验证集来自训练子集，非独立 dev 集） | 基线在独立 dev 集上评估 |
| 8 | 03-fasttext/api.py | 模型路径硬编码到讲师目录；`res[0][0][9:]` 魔法数字截断 `__label__` | 路径走配置；label 解析用前缀剥离 |
| 9 | 场景口径 | 新闻标题分类（THUCNews） | 物流寄件托寄物文本分类（简历口径），类目/数据/文案全部对齐 |
| 10 | 密钥 | `APIKEY.env` 含真实 API Key 且在工作区内 | 仅保留 `.env.example`，真实密钥不入库 |

另：fasttext 0.9.x 与 numpy 2.x 不兼容（`np.array(probs, copy=False)`），已在运行环境补丁为
`np.asarray`；requirements 未锁 numpy 降级（torch 2.11 / transformers 5.17 依赖 numpy 2.x）。
