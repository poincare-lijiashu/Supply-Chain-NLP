# -*- coding: utf-8 -*-
"""知识蒸馏：软标签（KL·T²，T=2.0, α=0.7）/ 硬标签 / 中间层（MSE）三种方式训练 BiLSTM 学生。"""
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from tqdm import tqdm
from transformers import BertConfig

from config.config import Config
from src.bert_model.bert_pipeline import BertClassifier, build_loaders, evaluate
from src.compress.bilstm import BiLSTMClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def soft_distill_loss(s_logits: torch.Tensor, t_logits: torch.Tensor, y: torch.Tensor,
                      T: float, alpha: float, ce: nn.Module | None = None) -> torch.Tensor:
    """软标签蒸馏损失（纯函数，便于单测）：α·KL(softmax(s/T)‖softmax(t/T))·T² + (1-α)·CE(s, y)。"""
    ce = ce if ce is not None else nn.CrossEntropyLoss()
    soft = F.kl_div(F.log_softmax(s_logits / T, dim=1),
                    F.softmax(t_logits / T, dim=1),
                    reduction="batchmean") * (T * T)
    return alpha * soft + (1 - alpha) * ce(s_logits, y)


@torch.no_grad()
def evaluate_student(model, loader, conf, device=None):
    device = device or conf.device
    model.eval()
    preds, labels = [], []
    for input_ids, attention_mask, y in loader:
        input_ids, attention_mask, y = (input_ids.to(device), attention_mask.to(device), y.to(device))
        logits = model(input_ids, attention_mask)
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        labels.extend(y.cpu().numpy())
    return accuracy_score(labels, preds)


def distill(conf: Config, mode: str = "soft", limit: int | None = None):
    """mode: soft(软标签) / hard(硬标签) / intermediate(中间层)"""
    assert mode in ("soft", "hard", "intermediate")
    train_loader, dev_loader, test_loader = build_loaders(conf, limit=limit)

    teacher = BertClassifier(conf).to(conf.device)
    teacher.load_state_dict(torch.load(conf.model_save_path, map_location=conf.device,
                                       weights_only=True))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = BiLSTMClassifier(conf).to(conf.device)
    optimizer = AdamW(student.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    T, alpha = conf.distill_T, conf.distill_alpha
    save_path = conf.student_save_path.replace(".pt", f"_{mode}.pt")

    print(f"[蒸馏-{mode}] T={T}, alpha={alpha}, device={conf.device}, "
          f"batches={len(train_loader)}")
    for epoch in range(conf.num_epochs):
        student.train()
        total_loss = 0.0
        t0 = time.time()
        for input_ids, attention_mask, y in tqdm(
                train_loader, desc=f"distill-{mode} epoch {epoch + 1}/{conf.num_epochs}", leave=False):
            input_ids, attention_mask, y = (input_ids.to(conf.device),
                                            attention_mask.to(conf.device), y.to(conf.device))
            optimizer.zero_grad()
            with torch.no_grad():
                t_logits, t_hidden = teacher.forward_hidden(input_ids, attention_mask)
            if mode == "soft":
                s_logits = student(input_ids, attention_mask)
                loss = soft_distill_loss(s_logits, t_logits, y, T, alpha, ce)
            elif mode == "hard":
                s_logits = student(input_ids, attention_mask)
                hard_target = torch.argmax(t_logits, dim=1)
                loss = ce(s_logits, hard_target)
            else:  # intermediate
                s_logits, s_hidden = student(input_ids, attention_mask, return_hidden=True)
                loss = alpha * F.mse_loss(s_hidden, t_hidden) + (1 - alpha) * ce(s_logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        dev_acc = evaluate_student(student, dev_loader, conf)
        print(f"  loss={total_loss / len(train_loader):.4f} dev_acc={dev_acc:.4f} ({time.time() - t0:.0f}s)")
        torch.save(student.state_dict(), save_path)

    test_acc = evaluate_student(student, test_loader, conf)
    n_params = sum(p.numel() for p in student.parameters())
    size_mb = n_params * 4 / 1024 / 1024
    summary = (f"蒸馏-{mode}: Test Acc={test_acc:.4f}, "
               f"参数量={n_params:,}（约{size_mb:.1f}MB fp32 / checkpoint: {save_path}）")
    print(summary)
    return test_acc, summary


if __name__ == "__main__":
    conf = Config()
    conf.bert_config = BertConfig.from_pretrained(conf.pretrain_bert_dir)
    # v2 蒸馏开关：BERT_INIT=1 用 BERT 字向量初始化学生（P12b 核心改进）
    conf.init_from_bert = os.getenv("BERT_INIT", "0") == "1"
    # 模式：默认三个全跑；v2 可 DST_MODE=soft 只跑 soft（30万数据上跑三模式太重，soft 为主轨）
    modes = [m.strip() for m in os.getenv("DST_MODE", "soft,hard,intermediate").split(",") if m.strip()]
    print(f"[distill] data_version={conf.data_version}, init_from_bert={conf.init_from_bert}, modes={modes}")
    results = []
    for mode in modes:
        results.append(distill(conf, mode=mode)[1])
    out = os.path.join(PROJECT_ROOT, "experiments", "distill.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("# 知识蒸馏实验（教师 BERT -> 学生 BiLSTM）\n\n```\n" + "\n".join(results) + "\n```\n")
    print(f"已保存 -> {out}")
    # Windows 下 CUDA 上下文清理会触发 0xC0000409 崩溃（退出码非 0），
    # 导致任务框架自动重跑并覆盖已保存的 checkpoint。跳过清理强制正常退出。
    os._exit(0)
