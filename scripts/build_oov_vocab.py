# -*- coding: utf-8 -*-
"""构建训练语料词表（jieba 分词），供 serving OOV 护栏判定"模型没见过的词"。

输入: data/raw_v2/train.txt（30 万条，训练只读）
输出: data/processed/oov_vocab.pkl（set[str]）
跳过: 纯数字/纯标点/纯英文单词 token（不参与未登录词判定）
"""
import io
import os
import pickle
import re

import jieba

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "data", "raw_v2", "train.txt")
OUT = os.path.join(ROOT, "data", "processed", "oov_vocab.pkl")
_SKIP = re.compile(r"^[\d\W_]+$")   # 纯数字/标点
_ASCII = re.compile(r"^[A-Za-z]+$")  # 纯英文单词(字母)

vocab = set()
n = 0
with io.open(TRAIN, encoding="utf-8") as f:
    for line in f:
        t = line.rstrip("\n").split("\t", 1)[0].strip()
        if not t:
            continue
        n += 1
        for w in jieba.cut(t):
            w = w.strip()
            if not w or _SKIP.match(w) or _ASCII.match(w):
                continue
            if len(w) >= 2 or any("\u4e00" <= ch <= "\u9fff" for ch in w):
                vocab.add(w)
print("句子数 %d, 词表词元 %d" % (n, len(vocab)))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "wb") as fh:
    pickle.dump(vocab, fh, protocol=4)
print("已写:", OUT)