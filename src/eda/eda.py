# -*- coding: utf-8 -*-
"""数据探索分析：类目分布、文本长度统计（确定截断长度依据）。结果落盘 experiments/。"""
import os
from collections import Counter

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  （matplotlib 必须先设定后端再引 pyplot）

from config.config import Config

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def analyze(path: str, class_list: list, split: str, out_dir: str):
    df = pd.read_csv(path, sep="\t", header=None, names=["text", "label"])
    counts = Counter(df["label"])
    df["len"] = df["text"].str.len()

    lines = [f"## {split}（{len(df)} 条）", ""]
    lines.append("| 类目 | 条数 | 占比 |")
    lines.append("|---|---|---|")
    for idx, name in enumerate(class_list):
        c = counts.get(idx, 0)
        lines.append(f"| {idx} {name} | {c} | {c / len(df) * 100:.1f}% |")
    lines += [
        "",
        f"- 文本长度：均值 {df['len'].mean():.2f}，标准差 {df['len'].std():.2f}，"
        f"最大 {df['len'].max()}，最小 {df['len'].min()}，P95 {df['len'].quantile(0.95):.0f}",
        f"- 截断长度依据：P95={df['len'].quantile(0.95):.0f} 字符，BERT 输入取 max_len=32 覆盖绝大多数样本",
        "",
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [f"{i}\n{n}" for i, n in enumerate(class_list)]
    ax.bar(names, [counts.get(i, 0) for i in range(len(class_list))])
    ax.set_title(f"{split} 类目分布")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{split}_dist.png"), dpi=120)
    plt.close(fig)
    return "\n".join(lines)


def main():
    conf = Config()
    out_dir = os.path.join(PROJECT_ROOT, "experiments")
    os.makedirs(out_dir, exist_ok=True)
    report = ["# 数据探索报告（托寄物文本合成数据）", ""]
    for split, path in [("train", conf.train_path), ("dev", conf.dev_path), ("test", conf.test_path)]:
        report.append(analyze(path, conf.class_list, split, out_dir))
    report_path = os.path.join(out_dir, "eda_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"EDA 完成 -> {report_path}")
    print("\n".join(report[2:12]))


if __name__ == "__main__":
    main()
