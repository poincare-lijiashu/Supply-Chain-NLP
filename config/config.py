# -*- coding: utf-8 -*-
"""统一配置：路径全部相对项目根目录，跨平台；超参与项目验收基准一致。"""
import os
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def _p(*parts):
    path = os.path.join(PROJECT_ROOT, *parts)
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    return path


class Config:
    def __init__(self):
        # ---------- 数据 ----------
        self.train_path = _p("data", "raw", "train.txt")
        self.dev_path = _p("data", "raw", "dev.txt")
        self.test_path = _p("data", "raw", "test.txt")
        self.class_path = _p("data", "class.txt")
        self.stopwords_path = _p("data", "stopwords.txt")
        self.process_dir = _p("data", "processed")
        self.process_train_path = _p("data", "processed", "train_process.csv")
        self.process_dev_path = _p("data", "processed", "dev_process.csv")
        self.process_test_path = _p("data", "processed", "test_process.csv")
        self.fasttext_dir = _p("data", "processed", "fasttext")
        self.train_ft_char_path = _p("data", "processed", "fasttext", "train_ft_char.txt")
        self.dev_ft_char_path = _p("data", "processed", "fasttext", "dev_ft_char.txt")
        self.test_ft_char_path = _p("data", "processed", "fasttext", "test_ft_char.txt")
        self.train_ft_jieba_path = _p("data", "processed", "fasttext", "train_ft_jieba.txt")
        self.dev_ft_jieba_path = _p("data", "processed", "fasttext", "dev_ft_jieba.txt")
        self.test_ft_jieba_path = _p("data", "processed", "fasttext", "test_ft_jieba.txt")

        # ---------- 类目 ----------
        self.class_list = []
        with open(self.class_path, "r", encoding="utf-8") as f:
            self.class_list = [line.strip() for line in f if line.strip()]
        self.num_classes = len(self.class_list)

        # ---------- 设备 ----------
        # MODEL_DEVICE 环境变量可强制指定（如 Windows 系统内存吃紧时 WDDM 假 OOM，可 MODEL_DEVICE=cpu）
        self.device = torch.device(os.getenv("MODEL_DEVICE", "") or
                                   ("cuda" if torch.cuda.is_available() else "cpu"))

        # ---------- BERT 微调超参 ----------
        self.model_name = "bert"
        self.pretrain_bert_dir = _p("models", "bert-base-chinese")
        self.num_epochs = 4
        self.batch_size = 128
        self.learning_rate = 5e-5
        self.max_len = 32            # 托寄物短文本，P95 长度约 11 字符，32 足够覆盖
        self.hidden_size = 768
        self.model_save_dir = _p("models", "checkpoints")
        self.model_save_path = _p("models", "checkpoints", "bert_best.pt")

        # ---------- fasttext ----------
        self.ft_model_dir = _p("models", "fasttext")
        self.ft_autotune_duration = 300   # 秒，自动调参时长

        # ---------- 随机森林 ----------
        self.rf_model_dir = _p("models", "baseline_rf")
        self.rf_max_words = 20000         # 基线取 2 万条子集验证可行性

        # ---------- 蒸馏超参 ----------
        self.distill_T = 2.0              # 蒸馏温度
        self.distill_alpha = 0.7          # 软损失权重
        self.student_embed = 128
        self.student_hidden = 256
        self.student_layers = 2
        self.student_dropout = 0.3
        self.student_save_path = _p("models", "checkpoints", "distill_best.pt")

        # ---------- 剪枝 ----------
        self.prune_amount = 0.3           # L1 非结构化剪枝比例

        # ---------- 部署（FastAPI） ----------
        self.serve_host = "127.0.0.1"     # 默认仅本机；容器/生产由 CMD 或网关显式暴露
        self.serve_port = 8004
        self.confidence_threshold = 0.60  # 低置信度转人工复核（违禁品红线策略）

        # ---------- API 安全（鉴权见 serving/app.py 的 API_AUTH_KEY 环境变量） ----------
        self.api_max_batch = 100          # 单次请求最大条数
        self.api_max_text_chars = 512     # 单条文本最大字符数（模型截断 32 token，此处防滥用）
        self.api_rate_limit_per_min = 60  # 每客户端每分钟最大请求数（内存滑动窗口，单进程口径）

    def id2name(self, idx: int) -> str:
        return self.class_list[idx] if 0 <= idx < len(self.class_list) else "其他"
