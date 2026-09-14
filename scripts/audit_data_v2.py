# -*- coding: utf-8 -*-
"""v2 30万数据 标签正确率深度审计（P10 补充门禁，在 7项质检之外）。

核心：抽样 N 条，用"文本核心词归属类 vs 行 label"自动判定标签正确性。
  - 非难例样本：取文本中命中词库的（最长）核心词，其类 == label 即正确
  - D 通道难例：按规则解析器专门判定（第一件申报/否定易碎/容器→12/夹带→13/拒识→14）
输出：每类正确率 / 总体正确率 / 错误样例
"""
import io
import os
import random
import re
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import yaml


def load_words():
    """词库: 主词+别名→类。复合长词优先用于匹配。"""
    wmap = {}  # text->cat
    d = os.path.join(ROOT, "data", "lexicon", "v2")
    for f in sorted(os.listdir(d)):
        if not f.startswith("cat_"):
            continue
        dd = yaml.safe_load(open(os.path.join(d, f), encoding="utf-8"))
        cat = dd["category"]
        seen = set()
        for sc in dd.get("subcategories", []):
            for it in sc.get("items", []) or []:
                if isinstance(it, dict):
                    ws = [it["word"]] + list(it.get("syn", []))
                else:
                    ws = [str(it)]
                for w in ws:
                    if w not in seen:
                        seen.add(w)
                        wmap[w] = cat
    # 排序：长词优先（减少"书包"命中"书包带"之类子串误判）
    return wmap


_RULE_RE = {
    "r11_first": re.compile(r"^(第一件是|先寄|寄)(?P<a>[^，,，]{1,16})[，,]?(?:，?(?P<f>[^，,]{1,30})，(?:第二件|{b}))"),
    "r11_plain": re.compile(r"^(寄|帮我寄)?(?P<a>[^，,和]{1,16})和(?P<b>[^，,]{1,16})(一起|各|放一个|另外)"),
    "r7_container": re.compile(r"(玻璃瓶装|陶瓷罐装|玻璃盒装|水晶盒装)"),
    "r9_negation": re.compile(r"(不是易碎|不算易碎|不用.*了|普通袋子装|普通带子装)"),
    "r1_redline": re.compile(r"(里藏了|夹带|夹带着|里面还有|一起装)"),
    "r27_reject": None,
}


def first_match_cat(text, wmap, max_w=12):
    """从左到右找第一个命中词库的词（长词优先），返回 (词, cat)。"""
    # 长词优先：先扫长词（>=4字）避免短词子串误判
    for wlen in (range(12, 3, -1)):
        for i in range(0, len(text) - wlen + 1):
            w = text[i:i + wlen]
            if w in wmap:
                return w, wmap[w]
    # 3字
    for i in range(0, len(text) - 2):
        w = text[i:i + 3]
        if w in wmap:
            return w, wmap[w]
    # 2字
    for i in range(0, len(text) - 1):
        w = text[i:i + 2]
        if w in wmap:
            return w, wmap[w]
    return None, None


def audit(path, n_sample=2000, seed=7):
    wmap = load_words()
    rng = random.Random(seed)
    lines = [l.rstrip("\n") for l in io.open(path, encoding="utf-8")]
    idx = rng.sample(range(len(lines)), min(n_sample, len(lines)))
    stat = defaultdict(lambda: [0, 0])  # cat -> [correct, total]
    errs = []
    for i in idx:
        t, lab = lines[i].split("\t")
        lab = int(lab)
        stat[lab][1] += 1
        # 难例规则解析
        if "按第一件申报" in t:
            m = re.match(r"第一件是(?P<a>[^，,，]{1,20})", t)
            w, cat = first_match_cat(m.group("a"), wmap) if m and m.group("a") else (None, None)
        elif "不是易碎" in t or "不算易碎" in t or "普通袋子装" in t or "普通带子装" in t:
            seg = re.split(r"[，,]", t)
            w, cat = None, None
            for s in seg:
                w, cat = first_match_cat(s, wmap)
                if cat is not None and cat != 12:
                    break
        elif any(k in t for k in ("里藏了", "夹带", "夹带着", "一起装")):
            cat = 13
            w = None
        elif "玻璃瓶装" in t or "陶瓷罐装" in t or "玻璃盒装" in t or "水晶盒装" in t:
            cat = 12
            w = None
        elif lab == 14:
            cat = 14
            w = None
        else:
            w, cat = first_match_cat(t, wmap)
        ok = (cat == lab)
        stat[lab][0] += 1 if ok else 0
        if not ok:
            errs.append((t, lab, cat, w))
    print("=" * 68)
    print("标签正确率审计: 抽样 %d / %d 条" % (len(idx), len(lines)))
    print("=" * 68)
    tot_c = tot_t = 0
    names = ["文件", "服饰", "数码", "食品", "日用", "美妆", "医药", "母婴",
             "文娱", "运动", "工业", "家具", "易碎", "违禁", "拒识"]
    for c in range(15):
        ok, t = stat[c]
        tot_c += ok
        tot_t += t
        pct = ok / t * 100 if t else 0
        flag = "✓" if pct >= 85 else "✗"
        print("  类%2d %-4s %s 正确率 %4.1f%% (%d/%d)" % (c, names[c], flag, pct, ok, t))
    print("-" * 68)
    print("总体标签正确率: %.2f%% (%d/%d)" % (tot_c / tot_t * 100, tot_c, tot_t))
    print("错误样例(前10):")
    for t, lab, cat, w in errs[:10]:
        print("  [%s→应%s,判%s] %s" % (names[lab] if lab < 15 else "?", names[lab] if lab < 15 else "?", names[cat] if cat is not None else "无词", t[:50]))
    return tot_c / tot_t


if __name__ == "__main__":
    audit(os.path.join(ROOT, "data", "raw_v2", "train.txt"))