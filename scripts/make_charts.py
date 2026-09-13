# -*- coding: utf-8 -*-
"""生成 README 报告图表：模型阶梯对比 / 混淆矩阵热图 / 推理延迟对比。

用法（项目根目录）：python -m scripts.make_charts
产物：docs/img/model_stairs.png、bert_confusion_matrix.png、distill_confusion_matrix.png、latency.png
指标阶梯数据与本脚本内 REPORT 同步维护（来源：experiments/real_run.md 实测记录）。"""
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from config.config import Config

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(PROJECT_ROOT, "docs", "img")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 实测指标阶梯（与 experiments/real_run.md 一致；更新模型后请同步修改）
REPORT = {
    "labels": ["TF-IDF+随机森林\n(基线)", "fastText\n(字级 default)", "BERT 微调\n(教师)", "知识蒸馏 BiLSTM\n(上线模型)"],
    "acc": [0.8614, 0.8828, 0.9650, 0.9650],
    "colors": ["#9aa5b1", "#7f9ac4", "#3b6fb4", "#2f9e6e"],
}
LATENCY = {"labels": ["BERT fp32", "BERT int8", "蒸馏 BiLSTM"], "ms": [19.5, 13.2, 1.30]}


def model_stairs():
    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=150)
    bars = ax.bar(REPORT["labels"], REPORT["acc"], color=REPORT["colors"], width=0.55)
    for b, v in zip(bars, REPORT["acc"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v * 100:.2f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0.80, 1.0)
    ax.set_ylabel("测试集准确率")
    ax.set_title("模型升级路径：业务难度基准下的测试集准确率", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "model_stairs.png"))
    plt.close(fig)


def _confusion_heat(path: str, title: str, out: str):
    cm = np.loadtxt(path, delimiter=",", dtype=int)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    conf = Config()
    n = len(conf.class_list)
    fig, ax = plt.subplots(figsize=(7.6, 6.6), dpi=150)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)), ax.set_yticks(range(n))
    ax.set_xticklabels(conf.class_list, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(conf.class_list, fontsize=9)
    for i in range(n):
        for j in range(n):
            v = cm_norm[i, j]
            if v > 0.005:
                ax.text(j, i, f"{v * 100:.1f}", ha="center", va="center",
                        fontsize=7.5, color="white" if v > 0.5 else "#333")
    ax.set_xlabel("预测类目"), ax.set_ylabel("真实类目")
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, label="行归一化比例(%)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def latency_chart():
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=150)
    bars = ax.barh(LATENCY["labels"][::-1], LATENCY["ms"][::-1],
                   color=["#2f9e6e", "#3b6fb6", "#9aa5b1"], height=0.5)
    for b, v in zip(bars, LATENCY["ms"][::-1]):
        ax.text(v + 0.4, b.get_y() + b.get_height() / 2, f"{v:.2f} ms",
                va="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("CPU 单条推理延迟 (ms)")
    ax.set_title("压缩带来约 20× 推理加速", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "latency.png"))
    plt.close(fig)


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    model_stairs()
    exp = os.path.join(PROJECT_ROOT, "experiments")
    bert_cm = os.path.join(exp, "bert_confusion_matrix.csv")
    distill_cm = os.path.join(exp, "distill_confusion_matrix.csv")
    if os.path.exists(bert_cm):
        _confusion_heat(bert_cm, "BERT 教师 · 测试集混淆矩阵", os.path.join(IMG_DIR, "bert_confusion_matrix.png"))
    if os.path.exists(distill_cm):
        _confusion_heat(distill_cm, "蒸馏 BiLSTM（上线模型）· 测试集混淆矩阵",
                        os.path.join(IMG_DIR, "distill_confusion_matrix.png"))
    latency_chart()
    print(f"图表已生成 -> {IMG_DIR}")


if __name__ == "__main__":
    main()
