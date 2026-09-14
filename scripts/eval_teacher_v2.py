# -*- coding: utf-8 -*-
"""P12c · 教师 BERT 完整能力评测（六把评测尺 + 违禁召回 + battery）。

对已训练的 v2 教师模型（checkpoints_v2/bert_best.pt）做主流评测：
  - dev / test-ID：分布内 acc + 每类 P/R/F1 + 混淆矩阵
  - test-NEAR / test-FAR：近义与常识泛化（裸词）
  - test-REJECT：拒识准确率（14 类）
  - test-SIM：业务仿真 acc + 违禁召回率（一票否决指标）
  - battery：由裁决手册固定句式逐条实测（含单条≡批量）
"""
import os
import sys

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from transformers import BertTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("DATA_VERSION", "v2")
from config.config import Config
from src.bert_model.bert_pipeline import BertClassifier, build_loaders, evaluate


def load_model(conf):
    model = BertClassifier(conf).to(conf.device)
    model.load_state_dict(torch.load(conf.model_save_path, map_location=conf.device,
                                     weights_only=True))
    model.eval()
    return model


def evaluate_file(model, tok, conf, path):
    """通用：读 text\\tlabel 文件，逐条（模拟真实单条推理）出 acc + 每类混淆。"""
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t, c = line.rsplit("\t", 1)
            texts.append(t)
            labels.append(int(c))
    preds = []
    with torch.no_grad():
        for t in texts:
            enc = tok([t], add_special_tokens=True, padding=True, truncation=True,
                      max_length=conf.max_len, return_attention_mask=True, return_tensors="pt")
            logits = model(enc["input_ids"].to(conf.device),
                           enc["attention_mask"].to(conf.device))
            preds.append(int(torch.argmax(logits, dim=-1).item()))
    return labels, preds, texts


def main():
    conf = Config()
    model = load_model(conf)
    tok = BertTokenizer.from_pretrained(conf.pretrain_bert_dir)
    print("=" * 70)
    print("教师 BERT · v2 15类 · 能力评测（六尺子 + 违禁召回 + battery）")
    print("checkpoint:", conf.model_save_path)
    print("=" * 70)

    summary = {}
    for name in ["dev", "test-ID", "test-NEAR", "test-FAR", "test-REJECT", "test-SIM"]:
        path = os.path.join(ROOT, "data", "raw_v2", f"{name}.txt")
        if not os.path.exists(path):
            continue
        y, p, _ = evaluate_file(model, tok, conf, path)
        acc = accuracy_score(y, p)
        f1 = f1_score(y, p, average="macro", zero_division=0)
        summary[name] = (acc, f1)
        print(f"\n--- {name}: acc={acc:.4f}  macro-F1={f1:.4f}  (n={len(y)}) ---")
        if name in ("test-ID", "test-SIM"):
            # 每类召回（重点看违禁13）
            cm = confusion_matrix(y, p, labels=list(range(15)))
            for c in range(15):
                tp = cm[c][c]
                total = cm[c].sum()
                if total:
                    print(f"   类{c:<2} 召回 {tp/total:.3f}", end="  ")
                    if (c + 1) % 4 == 0:
                        print()
            print()

    # 违禁召回率（test-SIM 中 label=13 的召回，一票否决指标）
    if "test-SIM" in summary:
        y, p, _ = evaluate_file(model, tok, conf,
                                os.path.join(ROOT, "data", "raw_v2", "test-SIM.txt"))
        y, p = np.array(y), np.array(p)
        mask = y == 13
        if mask.sum():
            rec = (p[mask] == 13).mean()
            prec = (p == 13)[(p == 13)].sum() / max((p == 13).sum(), 1)
            print("\n--- 违禁召回（test-SIM，一票否决）---")
            print(f"  违禁13召回率: {rec:.4f}（门槛≥0.95，目标≥0.98）")
            print(f"  违禁精确率: {prec:.4f}（转人工成本口径）")

    # battery：裁决手册固定句式 + 单条≡批量
    print("\n--- battery（抽样 12 条） ---")
    battery = ["一箱车厘子", "西瓜", "胶带", "充电宝2个", "陶瓷碗4个轻拿轻放",
               "6204轴承10个", "一箱东西", "给小孩寄的奶粉", "脚皮", "散装白酒 5L",
               "席卷 2 台 iPhone15", "医用口罩50个"]
    for t in battery:
        enc = tok([t], add_special_tokens=True, padding=True, truncation=True,
                  max_length=conf.max_len, return_attention_mask=True, return_tensors="pt")
        with torch.no_grad():
            prob = torch.softmax(model(enc["input_ids"].to(conf.device),
                                       enc["attention_mask"].to(conf.device)), dim=-1)[0]
        v, i = prob.max(dim=-1)
        print(f"  {t:<18} -> {conf.class_list[i]} {v:.2%}")

    print("\n评测完成")


if __name__ == "__main__":
    main()