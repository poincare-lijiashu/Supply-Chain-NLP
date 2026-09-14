# -*- coding: utf-8 -*-
"""P13教师能力·补充维度评测（主六尺之外的增长项）：

  A. 分布内细粒度：test-ID / test-SIM 的 每类 P/R/F1 + 混淆热区（top5 易混对）
  B. 违禁13 二分类口径：召回率/精确率/AUROC/PR-AUC（阈值无关，防 argmax 单点失真）
  C. 业务侧：转人工率@0.6（低置信度比例），违禁耦合人工复核后的净捕获率
  D. 置信度校准：ECE（10 桶，生产分布 test-SIM）
  E. 行为一致性：battery 单条≡批量；单条推理延迟 p50/p95（GPU 冷却预热后）
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("DATA_VERSION", "v2")
from config.config import Config
from src.bert_model.bert_pipeline import BertClassifier
from transformers import BertTokenizer

THR = 0.60  # 与 config.confidence_threshold 一致


def load_model(conf):
    m = BertClassifier(conf).to(conf.device)
    m.load_state_dict(torch.load(conf.model_save_path, map_location=conf.device,
                                 weights_only=True))
    m.eval()
    return m


def batch_probs(model, tok, conf, texts, bs=512):
    probs = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], add_special_tokens=True, padding=True,
                  truncation=True, max_length=conf.max_len,
                  return_attention_mask=True, return_tensors="pt")
        with torch.no_grad():
            lg = model(enc["input_ids"].to(conf.device),
                       enc["attention_mask"].to(conf.device))
        probs.append(torch.softmax(lg, dim=-1).cpu())
    return torch.cat(probs, dim=0)


def load_rows(name):
    path = os.path.join(ROOT, "data", "raw_v2", f"{name}.txt")
    rows = [l.rstrip("\n").split("\t") for l in io.open(path, encoding="utf-8")]
    return [r[0] for r in rows], [int(r[1]) for r in rows]


def inspect(name, texts, y, probs):
    names = ["文件", "服饰", "数码", "食品", "日用", "美妆", "医药", "母婴",
             "文娱", "运动", "工业", "家具", "易碎", "违禁", "拒识"]
    preds = probs.argmax(-1).numpy()
    confs = probs.max(-1).values.numpy()
    low = confs < THR
    acc = accuracy_score(y, preds)
    mf1 = f1_score(y, preds, average="macro", zero_division=0)
    wf1 = f1_score(y, preds, average="weighted", zero_division=0)
    print(f"\n--- {name}: acc={acc:.4f} macroF1={mf1:.4f} weightedF1={wf1:.4f} (n={len(y)}) ---")
    p, r, f, _ = precision_recall_fscore_support(y, preds, labels=list(range(15)),
                                                 zero_division=0)
    for c in range(15):
        flag = "!" if c == 13 else " "
        bar = "█" * int(round(f[c] * 30))
        print(f"  类{c:<2}{names[c]:<4} P {p[c]:.3f} R {r[c]:.3f} F1 {f[c]:.3f} {flag}{bar}")
    cm = confusion_matrix(y, preds, labels=list(range(15)))
    off = []
    for a in range(15):
        for b in range(15):
            if a != b and cm[a][b] > 0:
                off.append((cm[a][b], a, b))
    off.sort(reverse=True)
    print("  混淆热区 top5:", " | ".join(f"{names[a]}→{names[b]}×{n}" for n, a, b in off[:5]))
    # 违禁13 二分类口径（无违禁样本的集跳过）
    mask = np.array(y) == 13
    if mask.sum():
        print("  违禁13(argmax): 召回 R=%.4f  精确 P=%.4f  F1=%.4f" % (
            (preds[mask] == 13).mean(), p[13], f[13]))
        auc = roc_auc_score(mask, probs[:, 13].numpy())
        ap = average_precision_score(mask, probs[:, 13].numpy())
        print("  违禁13(AUC): AUROC=%.4f  PR-AUC=%.4f" % (auc, ap))
        # 转人工率 + 违禁净捕获（13 → 拦截；低置信 → 转人工，人工兜底也算"拦住"）
        low_13 = low & mask
        recall_policy = ((preds[mask] == 13) | low_13[mask]).mean()
        print("  转人工率@0.6: %.4f (%.0f/%d)  违禁净捕获率(含人工兜底): %.4f" %
              (low.mean(), low.sum(), len(y), recall_policy))
    else:
        auc = ap = float("nan")
        print("  违禁13: 本集无违禁样本，跳过")
    # 错判样本（小集/低准确率时打印头部，便于人工核因）
    if len(y) <= 2000 and acc < 0.99 and len(y) > 0:
        err = np.where(np.array(y) != preds)[0]
        print("  错判样本(前20):")
        for k in err[:20]:
            print("    [真%s→判%s conf=%.3f] %s" % (names[y[k]], names[preds[k]], confs[k], texts[k][:44]))
    # 校准 ECE
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(confs, bins) - 1
    ece = 0.0
    for b in range(10):
        m = idx == b
        if m.sum() == 0:
            continue
        ece += np.abs((confs[m].mean() - (preds[m] == np.array(y)[m]).mean())) * m.mean()
    print("  校准 ECE(10桶): %.4f" % ece)
    return acc, mf1, wf1, p[13], r[13], auc, ap, low.mean(), ece


def main():
    conf = Config()
    model = load_model(conf)
    tok = BertTokenizer.from_pretrained(conf.pretrain_bert_dir)
    print("教师补充维度评测, checkpoint:", conf.model_save_path)
    targets = sys.argv[1:] or ["test-ID", "test-SIM"]
    for name in targets:
        texts, y = load_rows(name)
        probs = batch_probs(model, tok, conf, texts)
        inspect(name, texts, y, probs)

    # battery: 单条≡批量 + 延迟
    battery = ["一箱车厘子", "西瓜", "胶带", "充电宝2个", "陶瓷碗4个轻拿轻放",
               "6204轴承10个", "一箱东西", "给小孩寄的奶粉", "脚皮", "散装白酒 5L",
               "席卷 2 台 iPhone15", "医用口罩50个"]
    print("\n--- battery: 单条≡批量一致性 ---")
    same = 0
    for t in battery:
        s = batch_probs(model, tok, conf, [t])
        b = batch_probs(model, tok, conf, [t] * 2)[0]
        a1 = int(s.argmax()); a2 = int(b.argmax())
        eq = a1 == a2 and bool(torch.allclose(s, b, atol=1e-4))
        same += eq
        print(f"  {t:<18} 单条={conf.class_list[a1]}({s.max():.2%}) "
              f"批量={conf.class_list[a2]} {'一致' if eq else '!不一致'}")
    print(f"  单条≡批量: {same}/{len(battery)}")
    # 延迟：GPU 预热后 200 次单条
    with torch.no_grad():
        for _ in range(5):
            enc = tok(["陶瓷碗4个轻拿轻放"], return_tensors="pt")
            model(enc["input_ids"].to(conf.device), enc["attention_mask"].to(conf.device))
    ts = []
    enc = tok(["陶瓷碗4个轻拿轻放"], add_special_tokens=True, padding=True,
              truncation=True, max_length=conf.max_len,
              return_attention_mask=True, return_tensors="pt")
    for _ in range(200):
        t0 = time.perf_counter()
        with torch.no_grad():
            model(enc["input_ids"].to(conf.device), enc["attention_mask"].to(conf.device))
        ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    print("\n--- 单条推理延迟 (GPU 预热后, n=200) ---")
    print("  p50=%.2fms  p95=%.2fms  mean=%.2fms" % (np.percentile(ts, 50),
                                                     np.percentile(ts, 95), ts.mean()))
    print("评测完成")


if __name__ == "__main__":
    main()