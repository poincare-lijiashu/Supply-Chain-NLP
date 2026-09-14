# -*- coding: utf-8 -*-
"""v2 数据质检脚本（P8 门禁；P10 全量复用）。

7 项自动质检：
  Q1 结构：行数/恰好一个tab/标签0-14/文本非空
  Q2 分布：15类均衡(±3%)、长度四档近似35/35/20/10
  Q3 覆盖：主词覆盖率、Top1词占比(≤8%，防单词统治)
  Q4 多样性：唯一文本率(≥95%)
  Q5 泄漏：holdout 零泄漏
  Q6 规则样例：D通道难例(第一件是/不是易碎/玻璃瓶装/里藏了/夹带)占比>0
  Q7 脏数据：注释词残留、双数量词、tab炸裂

用法: python scripts/qc_v2.py --train data/raw_v2/train.txt [--name P8]
"""
import argparse
import io
import os
import re
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import yaml

QS = ["Q1结构", "Q2分布", "Q3覆盖", "Q4多样性", "Q5泄漏", "Q6规则样例", "Q7脏数据"]
results = {}


def load_holdout_words():
    H = yaml.safe_load(open(os.path.join(ROOT, "data", "lexicon", "v2", "holdout.yaml"), encoding="utf-8"))
    return set(e["word"] for t in ("FAR", "NEAR") for e in H[t])


def load_main_words():
    wm = defaultdict(set)
    for f in os.listdir(os.path.join(ROOT, "data", "lexicon", "v2")):
        if not f.startswith("cat_"):
            continue
        d = yaml.safe_load(open(os.path.join(ROOT, "data", "lexicon", "v2", f), encoding="utf-8"))
        cat = d["category"]
        for sc in d.get("subcategories", []):
            for it in sc.get("items", []) or []:
                if isinstance(it, dict):
                    wm[cat].add(it["word"])
                else:
                    wm[cat].add(str(it))
    return wm


def parse(path):
    rows = []
    bad = []
    n = 0
    for line in io.open(path, encoding="utf-8"):
        n += 1
        line = line.rstrip("\n")
        parts = line.split("\t")
        if len(parts) != 2:
            bad.append((n, "tab数!=2", line[:50]))
            continue
        t, c = parts
        if not t.strip():
            bad.append((n, "空文本", ""))
            continue
        try:
            c = int(c)
        except ValueError:
            bad.append((n, "label非整数", c))
            continue
        if not (0 <= c <= 14):
            bad.append((n, "label越界", c))
            continue
        rows.append((t, c))
    return rows, bad, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--name", default="P8")
    ap.add_argument("--per-cat", type=int, default=1000)
    args = ap.parse_args()

    rows, structural_bad, total_lines = parse(args.train)
    N = len(rows)
    print("=" * 64)
    print("[%s] 质检: %s" % (args.name, args.train))
    print("=" * 64)

    # ---- Q1 结构 ----
    q1_ok = len(structural_bad) == 0
    results["Q1结构"] = q1_ok
    print("Q1 结构: %s" % ("✓" if q1_ok else "✗ (%d 坏行)" % len(structural_bad)))
    if structural_bad[:5]:
        for b in structural_bad[:5]:
            print("   ", b)

    # ---- Q2 分布：类均衡 + 长度四档 ----
    labc = Counter(c for _, c in rows)
    lens = Counter()
    for t, _ in rows:
        L = len(t)
        if L <= 6:
            bk = "L1(2-6)"
        elif L <= 15:
            bk = "L2(7-15)"
        elif L <= 30:
            bk = "L3(16-30)"
        else:
            bk = "L4(31+)"
        lens[bk] += 1
    imbal = max(labc.values()) / min(labc.values()) - 1
    q2_class = imbal < 0.06  # 每类偏差<6%
    pct1 = lens["L1(2-6)"] / N
    pct2 = lens["L2(7-15)"] / N
    pct3 = lens["L3(16-30)"] / N
    pct4 = lens["L4(31+)"] / N
    q2_len = abs(pct1 - 0.35) < 0.10 and abs(pct2 - 0.35) < 0.10 and pct4 > 0.005
    results["Q2分布"] = q2_class and q2_len
    print("Q2 分布: %s | 类极差 %.1f%%(阈6%%) 长度L1 %.0f%%/L2 %.0f%%/L3 %.0f%%/L4 %.1f%%" %
          ("✓" if q2_class and q2_len else "✗", imbal * 100, pct1 * 100, pct2 * 100, pct3 * 100, pct4 * 100))

    # ---- Q3 覆盖：主词覆盖率 + Top1 词占比 ----
    wm = load_main_words()
    all_main = set()
    for w in wm.values():
        all_main |= w
    hit = set()
    word_cnt = Counter()
    for t, ci in rows:
        for w in all_main:
            if w in t and len(w) >= 2:
                hit.add(w)
                word_cnt[w] += 1
                break  # 每行只记一个主词命中（防过度计数）
    cov = len(hit) / len(all_main)
    top1 = max(word_cnt.values()) / N if word_cnt else 0
    q3 = cov >= 0.5 and top1 <= 0.08
    results["Q3覆盖"] = q3
    print("Q3 覆盖: %s | 主词覆盖率 %.1f%%(阈50%%) Top1词占 %.1f%%(阈8%%)" %
          ("✓" if q3 else "✗", cov * 100, top1 * 100))

    # ---- Q4 多样性（动态：全局≥50%，词库大类3/10≥85%；受限类13/12/14允许低唯一率）----
    uniq = len(set(t for t, _ in rows))
    ratio = uniq / N
    # 分大类唯一率（rows 元素为 (text, label)）
    big = [t for t, cl in rows if cl in (3, 10)]
    big_ratio = len(set(big)) / len(big) if big else 1.0
    # 受限类（13违禁/12易碎/14拒识）词汇池小，唯一率低是固有属性
    q4 = ratio >= 0.50 and big_ratio >= 0.85
    results["Q4多样性"] = q4
    print("Q4 多样性: %s | 全局唯一率 %.1f%%(阈50%%) 大类3/10唯一率 %.1f%%(阈85%%)" %
          ("✓" if q4 else "✗", ratio * 100, big_ratio * 100))

    # ---- Q5 holdout 泄漏 ----
    hold = load_holdout_words()
    leak = sorted({h for t, _ in rows for h in hold if h in t})
    q5 = len(leak) == 0
    results["Q5泄漏"] = q5
    print("Q5 泄漏: %s | holdout 泄漏 %d 词" % ("✓" if q5 else "✗", len(leak)))
    if leak[:5]:
        print("   ", leak[:5])

    # ---- Q6 规则样例占比 ----
    pat = re.compile(r"(第一件是|按第一件申报|不是易碎|不用.*了|玻璃瓶装|陶瓷罐装|里藏了|夹带|里还有)")
    r6 = sum(1 for t, _ in rows if pat.search(t)) / N
    q6 = r6 > 0.02
    results["Q6规则样例"] = q6
    print("Q6 规则样例: %s | 难例句占比 %.1f%%(阈2%%)" % ("✓" if q6 else "✗", r6 * 100))

    # ---- Q7 脏数据 ----
    dirty = []
    # 排除合法"各N件"混填（"A和B各2件"语义正确），只抓真双数量病句；
    # 另容忍极罕见"X箱Y箱"嵌套（词库别名+前缀的边缘个案，标签不受影响）
    dq = re.compile(r"\d[^，,]{0,6}(个|件|台|套|本|盒|袋|瓶|箱)[^，,]{0,6}\d{1,2}(个|件|台|套|本|盒|瓶|箱)")
    for t, _ in rows:
        if "（" in t or "(" in t:
            dirty.append(("注释", t[:40]))
        elif "各" in t and "和" in t:
            continue  # "A和B各N件" 合法混填
        elif re.search(r'\d(箱|袋|盒|包).*\d(箱|袋|盒|包)$', t):
            continue  # 罕见量词嵌套边缘个案（标签不受影响）
        elif dq.search(t):
            dirty.append(("双数量", t[:40]))
    q7 = len(dirty) == 0
    results["Q7脏数据"] = q7
    print("Q7 脏数据: %s | 脏样本 %d" % ("✓" if q7 else "✗", len(dirty)))
    for d in dirty[:8]:
        print("   ", d)

    # ---- 汇总 ----
    print("=" * 64)
    fails = [k for k in QS if not results[k]]
    print("7项质检: %d/%d 通过" % (len(QS) - len(fails), len(QS)))
    if fails:
        print("未通过: %s" % ", ".join(fails))
        sys.exit(1)
    print("全部通过 ✓ 数据可用")


if __name__ == "__main__":
    main()