# -*- coding: utf-8 -*-
"""P13 复核工具：把 audit_data_v2 判"错"的抽样样本逐条摊开，
附上标准化的候选类命中线索，供人工区分【真实标签错误】 vs 【审计脚本缺陷】。

判定线索说明：
  - lab_hit : 文本做错别字反转(TYPO_REV)+去标点后，命中 label 类的词形（有=大概率脚本缺陷）
  - other    : 命中其它类的词形（多品混填/容器类时出现，助判）
"""
import io
import os
import random
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "data"))

import audit_data_v2 as A
from data.generate_v2 import Lexicon, _TYPO

TYPO_REV = {v: k for k, v in _TYPO.items()}


def norm(text):
    t = re.sub(r"[\s，,。.!！?？、；;：:（）()\"'“”·~—\-]+", "", text)
    for v, k in TYPO_REV.items():
        t = t.replace(v, k)
    return t


def main():
    lex = Lexicon()
    final = {c: set(lex.get_final_words(c)) for c in range(15)}
    wmap = A.load_words()

    rows = [l.rstrip("\n") for l in io.open(os.path.join(ROOT, "data", "raw_v2", "train.txt"), encoding="utf-8")]
    rng = random.Random(7)
    idx = rng.sample(range(len(rows)), 2000)

    mis = []
    for i in idx:
        t, lab_s = rows[i].split("\t")
        lab = int(lab_s)
        # == 与 audit_data_v2.audit 完全一致的分支，保证分歧集一致 ==
        if "按第一件申报" in t:
            m = re.match(r"第一件是(?P<a>[^，,，]{1,20})", t)
            w, cat = A.first_match_cat(m.group("a"), wmap) if m and m.group("a") else (None, None)
        elif any(k in t for k in ("不是易碎", "不算易碎", "普通袋子装", "普通带子装")):
            seg = re.split(r"[，,]", t)
            w, cat = None, None
            for s in seg:
                w, cat = A.first_match_cat(s, wmap)
                if cat is not None and cat != 12:
                    break
        elif any(k in t for k in ("里藏了", "夹带", "夹带着", "一起装")):
            cat, w = 13, None
        elif "玻璃瓶装" in t or "陶瓷罐装" in t or "玻璃盒装" in t or "水晶盒装" in t:
            cat, w = 12, None
        elif lab == 14:
            cat, w = 14, None
        else:
            w, cat = A.first_match_cat(t, wmap)
        if cat == lab:
            continue
        tn = norm(t)
        lab_hit = sorted({x for x in final[lab] if x and x in tn}, key=len, reverse=True)
        other = {c: sorted({x for x in final[c] if x and x in tn}, key=len, reverse=True)
                 for c in range(15) if c != lab and any(x in tn for x in final[c])}
        mis.append((t, lab, cat, w, lab_hit, other))

    print("分歧样本 %d 条 / 2000（= audit_data_v2 判错数）" % len(mis))
    stat = Counter()
    for t, lab, cat, w, lab_hit, other in mis:
        if lab_hit:
            stat["label类是命中(疑似脚本缺陷)"] += 1
        elif other:
            stat["label类无命中但他类命中(需求目判:混填/歧义)"] += 1
        else:
            stat["全无命中(需求目判:噪声/别名/OOV)"] += 1
    for k, v in stat.items():
        print("  %-30s %d" % (k, v))
    print("=" * 76)
    names = ["文件", "服饰", "数码", "食品", "日用", "美妆", "医药", "母婴",
             "文娱", "运动", "工业", "家具", "易碎", "违禁", "拒识"]
    for j, (t, lab, cat, w, lab_hit, other) in enumerate(mis):
        print("[%03d] label=%02d(%s) 判=%s 首词=%s" % (j, lab, names[lab], cat, w))
        print("  文本: %s" % t)
        if lab_hit:
            print("  lab_hit: %s" % lab_hit[:3])
        if other:
            print("  other  : %s" % {k: v[:2] for k, v in list(other.items())[:3]})
        if j >= 119:
            print("...(已输出前120条, 共%d)" % len(mis))
            break


if __name__ == "__main__":
    main()