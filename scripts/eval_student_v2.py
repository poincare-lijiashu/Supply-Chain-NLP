# -*- coding: utf-8 -*-
"""P14 蒸馏学生(BiLSTM)效果评测：六尺 + 违禁13 + 每类P/R/F1 + 单条≡批量 + 延迟 + 转人工率。

与教师(bert_best.pt)同口径对比：学生加载 models/checkpoints_v2/distill_best_soft.pt
"""
import io
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)
from transformers import BertTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("DATA_VERSION", "v2")
from config.config import Config
from src.compress.bilstm import BiLSTMClassifier

THR = float(os.getenv("OOV_LOW", "0.60")[:4])  # 与 serving 转人工阈值一致
TEACHER = {  # 教师已评测结果（P13 六尺），作对照
    "dev": 0.9995, "test-ID": 0.9996, "test-NEAR": 0.7556, "test-FAR": 1.0000,
    "test-REJECT": 1.0000, "test-SIM": 0.9972, "sim_forbid_recall": 1.0000,
    "latency_p50_ms": 4.82}


def load_model(conf):
    cp = conf.student_save_path.replace(".pt", "_soft.pt")
    m = BiLSTMClassifier(conf).to(conf.device)
    m.load_state_dict(torch.load(cp, map_location=conf.device, weights_only=True))
    m.eval()
    return m


def single_probs(model, tok, conf, text):
    """单条请求形状：返回 (logits, probs)"""
    enc = tok([text], add_special_tokens=True, padding=True, truncation=True,
              max_length=conf.max_len, return_attention_mask=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(enc["input_ids"].to(conf.device),
                       enc["attention_mask"].to(conf.device))
    return torch.softmax(logits, dim=-1)[0]


def eval_file(model, tok, conf, name):
    paths = os.path.join(ROOT, "data", "raw_v2", f"{name}.txt")
    texts, y = [], []
    for line in io.open(paths, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        t, c = line.rsplit("\t", 1)
        texts.append(t)
        y.append(int(c))
    confs_, probs_ = [], []
    for t in texts:  # 逐条 = 上线请求形状
        p = single_probs(model, tok, conf, t)
        confs_.append(float(p.max()))
        probs_.append(p.numpy() if p.device.type == "cpu" else p.cpu().numpy())
    y = np.array(y)
    preds = np.argmax(probs_, axis=1)
    acc = accuracy_score(y, preds)
    mf1 = f1_score(y, preds, average="macro", zero_division=0)
    return y, preds, np.array(confs_), np.array(probs_), acc, mf1, texts


def main():
    conf = Config()
    model = load_model(conf)
    tok = BertTokenizer.from_pretrained(conf.pretrain_bert_dir)
    n_params = sum(p.numel() for p in model.parameters())
    print("=" * 72)
    print("学生 BiLSTM 蒸馏效果评测（蒸馏自 v2 教师）")
    print("checkpoint: distill_best_soft.pt | 参数量 %d (%.2fMB)" % (n_params, n_params * 4 / 1024 / 1024))
    print("=" * 72)

    names = ["文件", "服饰", "数码", "食品", "日用", "美妆", "医药", "母婴",
             "文娱", "运动", "工业", "家具", "易碎", "违禁", "拒识"]
    for rk in ["dev", "test-ID", "test-NEAR", "test-FAR", "test-REJECT", "test-SIM"]:
        y, preds, confs, probs, acc, mf1, _ = eval_file(model, tok, conf, rk)
        tacc = TEACHER.get(rk)
        print(f"{rk:<10} 学生 acc={acc:.4f} F1macro={mf1:.4f} (n={len(y)})  | 教师 {tacc if tacc else '-'} "
              f"{'↓%.4f' % (acc - tacc) if tacc else ''}")
        if rk in ("test-ID", "test-SIM"):
            p_, r_, f_, _ = precision_recall_fscore_support(y, preds, labels=list(range(15)),
                                                            zero_division=0)
            print("        每类: " + " ".join(f"{names[c]}{r_[c]:.2f}" for c in range(15)))
        if rk == "test-SIM":
            mask = y == 13
            rec = (preds[mask] == 13).mean()
            prec = np.mean(y[preds == 13] == 13) if (preds == 13).any() else 0.0
            auc = roc_auc_score(mask, probs[:, 13]) if mask.sum() else float("nan")
            ap = average_precision_score(mask, probs[:, 13]) if mask.sum() else float("nan")
            low = confs < THR
            print(f"        违禁13: 召回 {rec:.4f} 精确 {prec:.4f} AUROC {auc:.4f} PR-AUC {ap:.4f} "
                  f"(教师召回 {TEACHER['sim_forbid_recall']:.4f}) | 转人工率 {low.mean():.4f}")

    # battery：单条≡批量 + 延迟
    battery = ["一箱车厘子", "西瓜", "胶带", "充电宝2个", "陶瓷碗4个轻拿轻放",
               "6204轴承10个", "一箱东西", "给小孩寄的奶粉", "脚皮", "散装白酒 5L",
               "席卷 2 台 iPhone15", "医用口罩50个"]
    same = 0
    for t in battery:
        s = single_probs(model, tok, conf, t)
        enc = tok([t] * 2, add_special_tokens=True, padding=True, truncation=True,
                  max_length=conf.max_len, return_attention_mask=True, return_tensors="pt")
        with torch.no_grad():
            b = torch.softmax(model(enc["input_ids"].to(conf.device),
                                    enc["attention_mask"].to(conf.device)), dim=-1)[0]
        ok = int(s.argmax()) == int(b.argmax()) and bool(torch.allclose(s, b, atol=1e-4))
        same += ok
    print(f"battery 单条≡批量: {same}/{len(battery)}")

    # 单条延迟（GPU 预热后 200 次）
    for _ in range(5):
        single_probs(model, tok, conf, "陶瓷碗4个轻拿轻放")
    ts = []
    t0 = time.perf_counter()
    for _ in range(200):
        single_probs(model, tok, conf, "陶瓷碗4个轻拿轻放")
    ms = (time.perf_counter() - t0) / 200 * 1000
    ts = [ms] * 200  # 逐次计时开销大，用整体均值近似
    print(f"单条推理平均 {ms:.2f}ms/条 (教师 p50 {TEACHER['latency_p50_ms']:.2f}ms)")


if __name__ == "__main__":
    main()