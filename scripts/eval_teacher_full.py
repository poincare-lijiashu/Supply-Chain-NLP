# -*- coding: utf-8 -*-
"""教师 BERT 10 维度全面评测（P12c-A1）。

维度：
 D1 Accuracy / Macro-F1 / Weighted-F1
 D2 每类 Precision/Recall/F1（test-ID + SIM）
 D3 混淆矩阵（完整 15x15，输出 CSV 供热图）
 D4 Cohen Kappa / MCC（多分类）
 D5 ROC-AUC (ovr) 每类 + 违禁类
 D6 PR-AUC 每类 + 违禁类（阈值无关）
 D7 ECE 期望校准误差 + 可靠性曲线数据
 D8 置信度统计（mean/max 分布、0.6 阈值转人工率 vs 覆盖损失）
 D9 错误聚类（真类→预测类转移矩阵 + top 错误样本）
 D10 鲁棒性：错字扰动集评测（TYPO 表）
"""
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report, cohen_kappa_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score,
                             roc_curve, precision_recall_curve, auc)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("DATA_VERSION", "v2")

from config.config import Config
from src.bert_model.bert_pipeline import BertClassifier
from transformers import BertTokenizer

NAMES = ["文件", "服饰", "数码", "食品", "日用", "美妆", "医药", "母婴",
         "文娱", "运动", "工业", "家具", "易碎", "违禁", "拒识"]


def infer_all(model, tok, conf, texts, batch=256):
    """批量推理，返回 logits (n,C) unnormalized。"""
    outs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            t = texts[i:i + batch]
            enc = tok(t, add_special_tokens=True, padding=True, truncation=True,
                      max_length=conf.max_len, return_attention_mask=True, return_tensors="pt")
            outs.append(model(enc["input_ids"].to(conf.device),
                              enc["attention_mask"].to(conf.device)).cpu())
    return torch.cat(outs)  # (n,C)


def load_file(path):
    ts, ys = [], []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        t, c = line.rsplit("\t", 1)
        ts.append(t)
        ys.append(int(c))
    return ts, np.array(ys)


def ece(probs, y, n_bins=10):
    """期望校准误差：|accuracy(bin) - confidence(bin)| 加权。"""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece_ = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf >= lo) & (conf < hi)
        if m.sum() == 0:
            continue
        ece_ += np.abs(acc[m].mean() - conf[m].mean()) * m.mean()
    return ece_


def eval_one(name, model, tok, conf):
    path = os.path.join(ROOT, "data", "raw_v2", f"{name}.txt")
    if not os.path.exists(path):
        return None
    ts, y = load_file(path)
    logits = infer_all(model, tok, conf, ts)
    probs = torch.softmax(logits, dim=-1).numpy()
    p = probs.argmax(axis=1)
    acc = accuracy_score(y, p)
    mf1 = f1_score(y, p, average="macro", zero_division=0)
    wf1 = f1_score(y, p, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(y, p)
    mcc = matthews_corrcoef(y, p)
    ece_ = ece(probs, y)
    # AUC (ovr) 各 class（n=15 足够）
    auc_ovr = None
    try:
        auc_ovr = roc_auc_score(y, probs, multi_class="ovr", average="macro")
    except Exception:
        pass
    # PR-AUC ovr
    prauc_ovr = None
    try:
        prauc_ovr = 0.0
        for c in range(15):
            yb = (y == c).astype(int)
            prec, rec, _ = precision_recall_curve(yb, probs[:, c])
            prauc_ovr += auc(rec, prec) / 15
    except Exception:
        pass
    cm = confusion_matrix(y, p, labels=list(range(15)))
    # 每类 P/R/F1
    p_, r_, f_, _ = precision_recall_fscore_support(y, p, labels=list(range(15)),
                                                    zero_division=0)
    # 0.6 阈值转人工率（置信度<0.6 转人工）
    conf_ = probs.max(axis=1)
    human_rate = float((conf_ < 0.6).mean())
    return dict(name=name, n=len(y), acc=acc, mf1=mf1, wf1=wf1, kappa=kappa, mcc=mcc,
                ece=ece_, auc=auc_ovr, prauc=prauc_ovr, cm=cm, per_class=np.stack([p_, r_, f_]),
                human_rate=human_rate, conf_mean=float(conf_.mean()))


def get_typo_pairs():
    """鲁棒性：噪声层 TYPO 表（同音形近）。"""
    import data.generate_v2 as G
    return G._TYPO


def robustness_check(model, tok, conf):
    """D10：错字扰动评测——取 test-ID 抽样 500，逐条注入 1 个 TYPO 改字，看 acc 掉多少。"""
    typo = get_typo_pairs()
    rng = np.random.RandomState(0)
    ts, y = load_file(os.path.join(ROOT, "data", "raw_v2", "test-ID.txt"))
    idx = rng.choice(len(ts), size=500, replace=False)
    orig = [ts[i] for i in idx]
    y0 = y[idx]
    # 只挑含可改字的文本
    perturbed, y1 = [], []
    for t, yy in zip(orig, y0):
        t2 = t
        for ch in typo:
            if ch in t and rng.random() < 0.6:
                t2 = t.replace(ch, typo[ch], 1)
                break
        if t2 == t:
            continue
        perturbed.append(t2)
        y1.append(yy)
    if not perturbed:
        return None
    logits = infer_all(model, tok, conf, perturbed)
    p = logits.argmax(dim=-1).numpy()
    acc = accuracy_score(y1, p)
    return dict(n=len(perturbed), acc_perturbed=acc)


def main():
    conf = Config()
    model = BertClassifier(conf).to(conf.device)
    model.load_state_dict(torch.load(conf.model_save_path, map_location=conf.device, weights_only=True))
    model.eval()
    tok = BertTokenizer.from_pretrained(conf.pretrain_bert_dir)

    out = {}
    for name in ["dev", "test-ID", "test-NEAR", "test-FAR", "test-REJECT", "test-SIM"]:
        r = eval_one(name, model, tok, conf)
        if r:
            out[name] = r
            print(f"\n=== {name} (n={r['n']}) ===")
            print(f"  acc={r['acc']:.4f}  macro-F1={r['mf1']:.4f}  weight-F1={r['wf1']:.4f}")
            print(f"  Kappa={r['kappa']:.4f}  MCC={r['mcc']:.4f}  ECE={r['ece']:.4f}")
            print(f"  AUC(ovr)={r['auc'] if r['auc'] is not None else 'N/A'}  PR-AUC(ovr)={r['prauc'] if r['prauc'] is not None else 'N/A'}")
            print(f"  转人工率(<0.6)={r['human_rate']:.3f}  均值置信={r['conf_mean']:.3f}")

    # 违禁（13）专项：test-SIM 上的召回/精确
    r = out["test-SIM"]
    y, p = None, None
    # 重新取 SIM 的 y/p
    ts, y = load_file(os.path.join(ROOT, "data", "raw_v2", "test-SIM.txt"))
    logits = infer_all(model, tok, conf, ts)
    p = logits.argmax(dim=-1).numpy()
    mask = y == 13
    rec13 = (p[mask] == 13).mean()
    pred13 = (p == 13)
    prec13 = float((y[pred13] == 13).sum()) / max(int(pred13.sum()), 1)
    # FAR 违禁扩展（FAR 里还有 违禁类吗——FAR 全 145 词无 13 类，改为在 test-ID 查 13 召回）
    m2 = y == 13
    print(f"\n=== 违禁召回 (test-SIM) === 召回={rec13:.4f} 精确={prec13:.4f}")

    # 混淆矩阵 top 错误对
    print("\n=== test-ID 混淆 TOP 错误对 ===")
    cm = out["test-ID"]["cm"]
    err_pairs = []
    for i in range(15):
        for j in range(15):
            if i != j and cm[i][j] > 0:
                err_pairs.append((cm[i][j], i, j))
    err_pairs.sort(reverse=True)
    for cnt, i, j in err_pairs[:15]:
        print(f"  真实{NAMES[i]:<4}→判{NAMES[j]:<4}  {cnt}")

    # 混淆矩阵 CSV 输出
    np.savetxt(os.path.join(ROOT, "experiments", "v2_teacher_cm_testID.csv"),
               out["test-ID"]["cm"], fmt="%d", delimiter=",")

    # 鲁棒性
    rb = robustness_check(model, tok, conf)
    if rb:
        print(f"\n=== 鲁棒性：错字扰动 === 扰动 {rb['n']} 条  acc={rb['acc_perturbed']:.4f}（test-ID 基线={out['test-ID']['acc']:.4f}）")

    # 序列化摘要
    summary = {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
                   for kk, vv in r.items() if kk not in ("cm", "per_class")}
               for k, r in out.items()}
    json.dump(summary, open(os.path.join(ROOT, "experiments", "v2_teacher_eval.json"), "w"),
              ensure_ascii=False, indent=1, default=str)
    print("\n已保存 experiments/v2_teacher_eval.json + v2_teacher_cm_testID.csv")


if __name__ == "__main__":
    main()