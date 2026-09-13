# -*- coding: utf-8 -*-
"""RF 基线：jieba 分词 -> CSV；TF-IDF + RandomForest；评估。基线取 2 万条子集验证任务可行性与数据可分性。"""
import os
import pickle
import time

import jieba
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, precision_score, recall_score)

from config.config import Config

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cut(text: str, max_words: int = 30) -> str:
    return " ".join(jieba.lcut(text)[:max_words])


def preprocess(conf: Config, limit: int | None = None):
    os.makedirs(conf.process_dir, exist_ok=True)
    for split, in_path, out_path in [
        ("train", conf.train_path, conf.process_train_path),
        ("dev", conf.dev_path, conf.process_dev_path),
        ("test", conf.test_path, conf.process_test_path),
    ]:
        if os.path.exists(out_path):
            continue
        df = pd.read_csv(in_path, sep="\t", header=None, names=["text", "label"])
        if limit:
            df = df[:limit]
        df["words"] = df["text"].apply(cut)
        df.to_csv(out_path, index=False)
        print(f"{split} 预处理完成 -> {out_path}")


def train_and_eval(conf: Config, subset: int = 20000):
    preprocess(conf)
    train_df = pd.read_csv(conf.process_train_path)[:subset]
    dev_df = pd.read_csv(conf.process_dev_path)

    stopwords = open(conf.stopwords_path, encoding="utf-8").read().split()
    tfidf = TfidfVectorizer(stop_words=stopwords)
    features = tfidf.fit_transform(train_df["words"])
    labels = train_df["label"]

    model = RandomForestClassifier(n_jobs=-1, random_state=22)
    t0 = time.time()
    model.fit(features, labels)
    train_time = time.time() - t0

    dev_features = tfidf.transform(dev_df["words"])
    preds = model.predict(dev_features)
    acc = accuracy_score(dev_df["label"], preds)
    report = classification_report(dev_df["label"], preds, target_names=conf.class_list, zero_division=0)

    os.makedirs(conf.rf_model_dir, exist_ok=True)
    with open(os.path.join(conf.rf_model_dir, "rf_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(conf.rf_model_dir, "tf_idf_vectorizer.pkl"), "wb") as f:
        pickle.dump(tfidf, f)

    summary = (f"随机森林基线（{subset} 条训练子集, TF-IDF+jieba）\n"
               f"训练耗时: {train_time:.1f}s\nDev Acc: {acc:.4f}\n"
               f"Micro P/R/F1: {precision_score(dev_df['label'], preds, average='micro'):.4f} / "
               f"{recall_score(dev_df['label'], preds, average='micro'):.4f} / "
               f"{f1_score(dev_df['label'], preds, average='micro'):.4f}\n\n{report}")
    print(summary)
    return acc, summary


if __name__ == "__main__":
    acc, summary = train_and_eval(Config())
    out = os.path.join(PROJECT_ROOT, "experiments", "baseline_rf.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("# 基线模型实验记录\n\n```\n" + summary + "\n```\n")
    print(f"已保存 -> {out}")
