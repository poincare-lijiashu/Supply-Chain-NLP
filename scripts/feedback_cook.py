# -*- coding: utf-8 -*-
"""P3 真实回流清洗聚合：同文本多单多数票 → 正/负/歧义 三路分流 + 抽审清单 + 周报。

输入: data/feedback/*.jsonl（/feedback 接口逐条落盘）
输出: data/feedback/clean/
  - positive.jsonl   模型判 == 终判（无偏真实分布池，供回归/混训）
  - negative.jsonl   模型判 != 终判（badcase 驱动补词库/边界修正）
  - ambiguous.jsonl  终判平票/零信息（人工仲裁）
  - review.txt       高风险抽审清单（违禁相关负样本 top，每周人工过）

用法: python scripts/feedback_cook.py [--dir data/feedback] [--top 30]
"""
import argparse
import glob
import json
import os
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = ["文件", "服饰", "数码", "食品", "日用", "美妆", "医药", "母婴",
         "文娱", "运动", "工业", "家具", "易碎", "违禁", "拒识"]


def clean_text(t: str) -> str:
    """零信息过滤：全符号/纯数字/纯空白。"""
    t = t.strip()
    if not t:
        return ""
    # 至少含一个中文字符或字母才认为有信息
    if not any("\u4e00" <= ch <= "\u9fff" or ch.isalpha() for ch in t):
        return ""
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    ap.add_argument("--top", type=int, default=30, help="抽审清单条数")
    args = ap.parse_args()
    fb_dir = args.dir or os.path.join(ROOT, "data", "feedback")

    rows = []
    for fn in sorted(glob.glob(os.path.join(fb_dir, "*.jsonl"))):
        with open(fn, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue

    # 按 text 聚合（同一托寄物描述 = 同一样本，多单 = 多数票）
    agg = defaultdict(list)
    for r in rows:
        r["text"] = (r.get("text") or "").strip()
        if clean_text(r["text"]):
            agg[r["text"]].append(r)

    positive, negative, ambiguous = [], [], []
    forb_neg = []
    n_rows_invalid = len(rows) - sum(len(v) for v in agg.values())

    for text, rs in agg.items():
        vc = Counter(r["human_class"] for r in rs)
        top = vc.most_common()
        # 平票（最高两票并列）→ 歧义，交给人工仲裁
        if len(top) > 1 and top[0][1] == top[1][1]:
            ambiguous.append({
                "text": text, "votes": dict(vc), "n": len(rs),
                "model_pred": rs[0].get("model_pred"), "model_conf": rs[0].get("model_conf"),
                "model_ver": rs[0].get("model_ver", ""),
                "changed": any(r.get("human_changed") for r in rs)})
            continue
        maj, maj_votes = top[0]
        m_pred = rs[0].get("model_pred")
        sample = {
            "text": text, "label": maj, "votes": dict(vc), "n": len(rs),
            "maj_votes": maj_votes,
            "model_pred": m_pred, "model_conf": rs[0].get("model_conf"),
            "model_ver": rs[0].get("model_ver", ""),
            "changed": any(r.get("human_changed") for r in rs),
        }
        if m_pred == maj:
            positive.append(sample)
        else:
            negative.append(sample)
            if 13 in vc or m_pred == 13:  # 涉违禁的负样本必须人工过
                forb_neg.append(sample)

    out = os.path.join(fb_dir, "clean")
    os.makedirs(out, exist_ok=True)

    def dump(name, data):
        with open(os.path.join(out, name), "w", encoding="utf-8") as f:
            for s in data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    dump("positive.jsonl", sorted(positive, key=lambda s: -s["model_conf"]))
    dump("negative.jsonl", sorted(negative, key=lambda s: -s["model_conf"]))
    dump("ambiguous.jsonl", ambiguous)

    # 高风险抽审清单（违禁相关负样本，缺省可复用）
    with open(os.path.join(out, "review.txt"), "w", encoding="utf-8") as f:
        f.write("### 负样本/风险抽审清单（每周人工过一遍，标注'确认/驳回'后回归）\n")
        todo = sorted(forb_neg, key=lambda s: -s["model_conf"])[:args.top]
        for i, s in enumerate(todo or negative[:args.top], 1):
            f.write("[%02d] mode=%s | 模型判%s(%.2f) | 终判%s(%d票/%d单) | %s\n" % (
                i, s["model_ver"], NAMES[s["model_pred"]] if s["model_pred"] is not None else "?",
                s["model_conf"] or 0, NAMES[s["label"]], s["maj_votes"], s["n"], s["text"]))

    # ---- 周报 ----
    uniq = len(agg)
    changed_rate = sum(1 for v in agg.values() if any(r.get("human_changed") for r in v)) / uniq if uniq else 0
    print("=" * 60)
    print("真实回流周报（%s）" % os.path.basename(fb_dir))
    print("=" * 60)
    print("原始回传: %d 条 | 无效(零信息)过滤: %d 条 | 唯一文本样本: %d" % (len(rows), n_rows_invalid, uniq))
    print("分流: 正样本 %d | 负样本(badcase) %d | 歧义 %d" % (len(positive), len(negative), len(ambiguous)))
    print("快递员改判率: %.1f%%" % (changed_rate * 100))
    print("涉违禁负样本: %d 条 → 已入 review.txt 抽审" % len(forb_neg))
    print("输出目录: %s" % out)
    if negative:
        print("\nTop5 badcase（按模型置信度排序，重点排查高置信错分）:")
        for s in sorted(negative, key=lambda x: -x["model_conf"])[:5]:
            print("  [模型判%s %.2f -> 快递员改%s] %s" % (
                NAMES[s["model_pred"]] if s["model_pred"] is not None else "?",
                s["model_conf"] or 0, NAMES[s["label"]], s["text"]))


if __name__ == "__main__":
    main()