# -*- coding: utf-8 -*-
"""fasttext 模块：字级/词级 × 默认/自动调参，评估并计时。"""
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
import fasttext  # noqa: E402
import jieba  # noqa: E402
from config.config import Config  # noqa: E402

fasttext.FastText.eprint = lambda x: None  # 关闭 fasttext 噪声日志


def preprocess(conf: Config):
    os.makedirs(conf.fasttext_dir, exist_ok=True)
    id2name = {i: n for i, n in enumerate(conf.class_list)}
    jobs = []
    for split, in_path in [("train", conf.train_path), ("dev", conf.dev_path), ("test", conf.test_path)]:
        for mode, out_path in [("char", os.path.join(conf.fasttext_dir, f"{split}_ft_char.txt")),
                               ("jieba", os.path.join(conf.fasttext_dir, f"{split}_ft_jieba.txt"))]:
            jobs.append((split, mode, in_path, out_path))
    for split, mode, in_path, out_path in jobs:
        if os.path.exists(out_path):
            continue
        with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                text, label = line.rsplit("\t", 1)
                label_name = f"__label__{id2name[int(label)]}"
                text = text.replace(":", "").replace("\t", " ")
                words = jieba.lcut(text) if mode == "jieba" else list(text)
                fout.write(f"{label_name} {' '.join(words)}\n")
        print(f"{split}/{mode} 格式转换完成 -> {out_path}")


def train(conf: Config, mode: str = "char", autotune: bool = True):
    preprocess(conf)
    train_path = conf.train_ft_char_path if mode == "char" else conf.train_ft_jieba_path
    dev_path = conf.dev_ft_char_path if mode == "char" else conf.dev_ft_jieba_path
    test_path = conf.test_ft_char_path if mode == "char" else conf.test_ft_jieba_path

    os.makedirs(conf.ft_model_dir, exist_ok=True)
    tag = f"{mode}_{'auto' if autotune else 'default'}"
    kwargs = dict(thread=1, verbose=2)
    if autotune:
        kwargs.update(autotuneValidationFile=dev_path, autotuneDuration=conf.ft_autotune_duration)
    print(f"[fasttext] 训练 {tag} ...")
    t0 = time.time()
    model = fasttext.train_supervised(input=train_path, **kwargs)
    train_time = time.time() - t0

    model_path = os.path.join(conf.ft_model_dir, f"fastText_{tag}.bin")
    model.save_model(model_path)

    res = model.test(test_path)
    # 单条预测耗时
    sample = " ".join(list("寄两份合同文件明天必须到"))
    t0 = time.time()
    for _ in range(100):
        model.predict(sample)
    latency_ms = (time.time() - t0) * 10

    summary = (f"fasttext[{tag}] 训练耗时 {train_time:.0f}s\n"
               f"Test: 样本数={res[0]}, P={res[1]:.4f}, R={res[2]:.4f}\n"
               f"单条预测耗时: {latency_ms:.2f} ms\n模型: {model_path}")
    print(summary)
    return summary


if __name__ == "__main__":
    conf = Config()
    results = []
    for mode in ["char", "jieba"]:
        results.append(train(conf, mode=mode, autotune=False))
        results.append(train(conf, mode=mode, autotune=True))
    out = os.path.join(PROJECT_ROOT, "experiments", "fasttext.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("# fasttext 四组实验（字/词 × 默认/自动调参）\n\n```\n" + "\n\n".join(results) + "\n```\n")
    print(f"已保存 -> {out}")
