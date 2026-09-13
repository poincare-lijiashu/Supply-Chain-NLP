# -*- coding: utf-8 -*-
"""一键模型评测：BERT 教师与蒸馏学生（上线口径）在测试集上的准确率 / Macro F1 / 混淆矩阵。

用法（项目根目录）：python -m scripts.eval_models [--model bert|distill|both]
结果落盘 experiments/bert_confusion_matrix.csv 与 experiments/distill_confusion_matrix.csv。"""
import argparse
import os

import numpy as np
import torch
from transformers import BertConfig

from config.config import Config
from src.bert_model.bert_pipeline import BertClassifier, build_loaders, evaluate
from src.compress.bilstm import BiLSTMClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _evaluate_and_save(model, test_loader, conf, tag: str) -> tuple[float, float]:
    acc, f1, report, cm, _, _ = evaluate(model, test_loader, conf)
    print(f"\n===== {tag} =====")
    print(f"Test Acc={acc:.4f}  Macro F1={f1:.4f}")
    print(report)
    if cm is not None:
        cm_path = os.path.join(PROJECT_ROOT, "experiments", f"{tag}_confusion_matrix.csv")
        np.savetxt(cm_path, cm, fmt="%d", delimiter=",")
        print(f"混淆矩阵 -> {cm_path}")
    return acc, f1


def main():
    parser = argparse.ArgumentParser(description="测试集模型评测")
    parser.add_argument("--model", choices=["bert", "distill", "both"], default="both")
    parser.add_argument("--batch", type=int, default=None, help="覆盖评估批大小")
    args = parser.parse_args()

    conf = Config()
    if args.batch:
        conf.batch_size = args.batch
    conf.bert_config = BertConfig.from_pretrained(conf.pretrain_bert_dir)
    _, _, test_loader = build_loaders(conf)
    print(f"device={conf.device}, test batches={len(test_loader)}")

    if args.model in ("bert", "both"):
        model = BertClassifier(conf).to(conf.device)
        model.load_state_dict(torch.load(conf.model_save_path, map_location=conf.device,
                                         weights_only=True))
        _evaluate_and_save(model, test_loader, conf, "bert")

    if args.model in ("distill", "both"):
        student = BiLSTMClassifier(conf).to(conf.device)
        student.load_state_dict(torch.load(
            conf.student_save_path.replace(".pt", "_soft.pt"),
            map_location=conf.device, weights_only=True))
        _evaluate_and_save(student, test_loader, conf, "distill")


if __name__ == "__main__":
    main()
