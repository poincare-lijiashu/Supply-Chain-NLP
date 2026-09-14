# -*- coding: utf-8 -*-
"""P9 基线：RF(TF-IDF+jieba) 与 fastText(default jieba/char / autotune 60s) 在 v2 15类上的分布内基线。

输入：data/raw_v2/{train,dev,test-ID}.txt（text\\tlabel，15类）
输出：控制台报告（acc + macro-F1 + 耗时），模型选存到 models/baseline_v2/。
P9 = 廉价信号阶段：只用 dev + test-ID 两把尺（P10全量后 P12 再六尺全报）。
"""
import io
import os
import sys
import time

import fasttext
import jieba
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw_v2")
OUT = os.path.join(ROOT, "models", "baseline_v2")
fasttext.FastText.eprint = lambda x: None  # 关 fasttext 日志


def load(split):
    rows = []
    with io.open(os.path.join(RAW, f"{split}.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t, c = line.rsplit("\t", 1)
            rows.append((t, int(c)))
    return rows


def cut(text):
    return " ".join(jieba.lcut(text))


def run_rf(train, dev, test):
    print("\n" + "=" * 60)
    print("[RF] TF-IDF(jieba) + RandomForest  15 类")
    Xtr = [cut(t) for t, _ in train]
    Xdv = [cut(t) for t, _ in dev]
    Xte = [cut(t) for t, _ in test]
    ytr = [c for _, c in train]
    ydv = [c for _, c in dev]
    yte = [c for _, c in test]

    tfidf = TfidfVectorizer(stop_words=[])
    Xtr_v = tfidf.fit_transform(Xtr)
    Xdv_v = tfidf.transform(Xdv)
    Xte_v = tfidf.transform(Xte)

    rf = RandomForestClassifier(n_jobs=-1, random_state=22)
    t0 = time.time()
    rf.fit(Xtr_v, ytr)
    t_train = time.time() - t0

    t0 = time.time()
    pdv = rf.predict(Xdv_v)
    pte = rf.predict(Xte_v)
    t_pred = time.time() - t0

    acc_dv = accuracy_score(ydv, pdv)
    acc_te = accuracy_score(yte, pte)
    f1_te = f1_score(yte, pte, average="macro", zero_division=0)
    print(f"  训练 {t_train:.1f}s | 预测 {t_pred:.2f}s")
    print(f"  acc:  dev={acc_dv:.4f}  test-ID={acc_te:.4f}  macro-F1(test)={f1_te:.4f}")
    os.makedirs(OUT, exist_ok=True)
    try:
        import joblib
        joblib.dump(rf, os.path.join(OUT, "rf_v2.pkl"))
        joblib.dump(tfidf, os.path.join(OUT, "tfidf_v2.pkl"))
        print(f"  模型已存 -> {OUT}/rf_v2.pkl")
    except Exception as e:
        print("  存模型跳过:", e)
    return acc_dv, acc_te


def _ft_subsample(train, cap=12000):
    """fastText 首轮用子集快速出数（P9 探路）；全量在 P12。"""
    if len(train) <= cap:
        return train
    import random
    rng = random.Random(7)
    rng.shuffle(train)
    return train[:cap]


def run_ft(train, dev, test, mode, autotune=False, duration=60):
    print("\n" + "=" * 60)
    tag = f"{mode}_{'auto' if autotune else 'default'}"
    print(f"[fastText] {tag}")
    os.makedirs(OUT, exist_ok=True)

    def to_ft(rows, path):
        with io.open(path, "w", encoding="utf-8") as f:
            for t, c in rows:
                toks = list(t) if mode == "char" else jieba.lcut(t)
                f.write("__label__%d %s\n" % (c, " ".join(toks)))

    tr = _ft_subsample(train)
    ft_tr, ft_dv, ft_te = os.path.join(OUT, f"p9_{tag}_tr.txt"), \
                          os.path.join(OUT, f"p9_{tag}_dv.txt"), \
                          os.path.join(OUT, f"p9_{tag}_te.txt")
    to_ft(tr, ft_tr)
    to_ft(dev, ft_dv)
    to_ft(test, ft_te)

    kwargs = dict(thread=1, verbose=0)
    if autotune:
        kwargs.update(autotuneValidationFile=ft_dv, autotuneDuration=duration)
    t0 = time.time()
    model = fasttext.train_supervised(input=ft_tr, **kwargs)
    t_train = time.time() - t0

    def acc_on(path):
        res = model.test(path)
        return res[1]  # precision == recall for test

    acc_dv = acc_on(ft_dv)
    acc_te = acc_on(ft_te)
    print(f"  训练 {t_train:.1f}s | acc: dev={acc_dv:.4f}  test-ID={acc_te:.4f}")
    try:
        model.save_model(os.path.join(OUT, f"fastText_v2_{tag}.bin"))
        print(f"  模型已存 -> {OUT}/fastText_v2_{tag}.bin")
    except Exception as e:
        print("  存模型跳过:", e)
    return acc_dv, acc_te


if __name__ == "__main__":
    print("P9 基线（v2 15 类 · 分布内 dev/test-ID）")
    train = load("train")
    dev = load("dev")
    test = load("test-ID")
    print(f"train={len(train)}  dev={len(dev)}  test-ID={len(test)}")

    results = {}
    results["RF"] = run_rf(train, dev, test)
    for mode in ("jieba", "char"):
        results[f"fastText_{mode}_default"] = run_ft(train, dev, test, mode, autotune=False)
    results["fastText_jieba_auto60"] = run_ft(train, dev, test, "jieba", autotune=True, duration=60)

    print("\n" + "=" * 60)
    print("P9 汇总（acc: dev / test-ID）")
    for k, (d, t) in results.items():
        print(f"  {k:<26} dev={d:.4f}  test-ID={t:.4f}")