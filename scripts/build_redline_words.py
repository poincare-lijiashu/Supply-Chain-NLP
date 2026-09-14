# -*- coding: utf-8 -*-
"""构建红线命中护栏词表：从 13 类（违禁限寄）词库提取主词+别名 → pkl。

serving 命中即强制转人工（与模型置信度无关），堵住"模型高置信错分到非违禁类"的漏网。
输出: data/processed/redline_words.json（list[str]，len>=2）
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.generate_v2 import Lexicon

words = {w for w in Lexicon().get_final_words(13) if len(w) >= 2}
out = os.path.join(ROOT, "data", "processed", "redline_words.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as fh:
    json.dump(sorted(words), fh, ensure_ascii=False)
print("红线词表: %d 词 -> %s" % (len(words), out))
# 抽检关键词
for probe in ("小猫", "充电宝", "散装白酒", "活体", "口罩"):
    print("  %s %s" % (probe, "命中" if probe in words else "-未命中"))