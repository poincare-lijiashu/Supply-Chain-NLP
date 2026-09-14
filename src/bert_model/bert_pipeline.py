# -*- coding: utf-8 -*-
"""BERT 文本分类：模型 / 数据 / 训练 / 评估 / 预测。相对初版实现修复三处缺陷：
1) 训练保存逻辑反向（验证集提升才保存）；2) predict 循环内误用整批 texts 编码；
3) evaluate 提前 break 只评一个 batch。"""
import gc
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertConfig, BertModel, BertTokenizer

from config.config import Config

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_raw_data(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text, label = line.rsplit("\t", 1)
            rows.append((text, int(label)))
    return rows


class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def build_loaders(conf: Config, limit: int | None = None):
    tokenizer = BertTokenizer.from_pretrained(conf.pretrain_bert_dir)

    def collate(batch):
        texts = [b[0] for b in batch]
        labels = [b[1] for b in batch]
        encoded = tokenizer(texts, add_special_tokens=True, padding=True,
                            truncation=True, max_length=conf.max_len,
                            return_attention_mask=True)
        return (torch.tensor(encoded["input_ids"]),
                torch.tensor(encoded["attention_mask"]),
                torch.tensor(labels))

    def make(path, shuffle):
        rows = load_raw_data(path)
        if limit:
            rows = rows[:limit]
        return DataLoader(TextDataset(rows), batch_size=conf.batch_size,
                          shuffle=shuffle, collate_fn=collate)

    return make(conf.train_path, True), make(conf.dev_path, False), make(conf.test_path, False)


class BertClassifier(nn.Module):
    def __init__(self, conf: Config):
        super().__init__()
        self.bert = BertModel.from_pretrained(conf.pretrain_bert_dir)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(conf.hidden_size, conf.num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids, attention_mask=attention_mask).pooler_output
        return self.fc(self.dropout(out))

    def forward_hidden(self, input_ids, attention_mask):
        """蒸馏用：返回 (logits, pooled_hidden[768])"""
        out = self.bert(input_ids, attention_mask=attention_mask).pooler_output
        return self.fc(self.dropout(out)), out


@torch.no_grad()
def evaluate(model, loader, conf: Config, device=None, full=True):
    device = device or conf.device
    model.eval()
    preds, labels = [], []
    for input_ids, attention_mask, y in tqdm(loader, desc="Evaluating", leave=False):
        input_ids, attention_mask, y = input_ids.to(device), attention_mask.to(device), y.to(device)
        logits = model(input_ids, attention_mask)
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        labels.extend(y.cpu().numpy())
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    report = classification_report(labels, preds, target_names=conf.class_list,
                                   zero_division=0, digits=4) if full else ""
    cm = confusion_matrix(labels, preds) if full else None
    return acc, f1, report, cm, preds, labels


def train(conf: Config, limit: int | None = None):
    train_loader, dev_loader, test_loader = build_loaders(conf, limit=limit)
    model = BertClassifier(conf).to(conf.device)
    optimizer = AdamW(model.parameters(), lr=conf.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    best_dev_acc = 0.0
    print(f"device={conf.device}, train batches={len(train_loader)}")
    for epoch in range(conf.num_epochs):
        model.train()
        total_loss, preds, labels = 0.0, [], []
        t0 = time.time()
        for step, (input_ids, attention_mask, y) in enumerate(
                tqdm(train_loader, desc=f"BERT epoch {epoch + 1}/{conf.num_epochs}")):
            input_ids, attention_mask, y = (input_ids.to(conf.device),
                                            attention_mask.to(conf.device), y.to(conf.device))
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(y.cpu().numpy())
        train_time = time.time() - t0

        dev_acc, dev_f1, _, _, _, _ = evaluate(model, dev_loader, conf, full=False)
        train_acc = accuracy_score(labels, preds)
        gc.collect()  # 长训练防宿主内存碎片（受限内存机器上 MemoryError 防护）
        torch.cuda.empty_cache()
        print(f"Epoch {epoch + 1}: loss={total_loss / len(train_loader):.4f} "
              f"train_acc={train_acc:.4f} dev_acc={dev_acc:.4f} dev_f1={dev_f1:.4f} "
              f"({train_time:.0f}s)")
        if dev_acc > best_dev_acc:  # 修复点：验证集提升才保存
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), conf.model_save_path)
            print(f"  保存最优模型 (dev_acc={dev_acc:.4f}) -> {conf.model_save_path}")
    return model, best_dev_acc, test_loader


def test_and_report(conf: Config, model=None):
    _, _, test_loader = build_loaders(conf)
    if model is None:
        model = BertClassifier(conf).to(conf.device)
        model.load_state_dict(torch.load(conf.model_save_path, map_location=conf.device,
                                         weights_only=True))
    t0 = time.time()
    acc, f1, report, cm, _, _ = evaluate(model, test_loader, conf)
    eval_time = time.time() - t0
    print(f"Test Acc: {acc:.4f}  Macro F1: {f1:.4f}  (评估 {eval_time:.0f}s)")
    print(report)
    np.savetxt(os.path.join(PROJECT_ROOT, "experiments", "bert_confusion_matrix.csv"),
               cm, fmt="%d", delimiter=",")
    return acc, f1, report, cm


def predict(texts, conf: Config, model_path: str | None = None, return_model=False):
    model_path = model_path or conf.model_save_path
    tokenizer = BertTokenizer.from_pretrained(conf.pretrain_bert_dir)
    model = BertClassifier(conf).to(conf.device)
    model.load_state_dict(torch.load(model_path, map_location=conf.device, weights_only=True))
    model.eval()
    if isinstance(texts, str):
        texts = [texts]
    encoded = tokenizer(texts, add_special_tokens=True, padding=True, truncation=True,
                        max_length=conf.max_len, return_attention_mask=True, return_tensors="pt")
    input_ids = encoded["input_ids"].to(conf.device)
    attention_mask = encoded["attention_mask"].to(conf.device)
    with torch.no_grad():
        probs = torch.softmax(model(input_ids, attention_mask), dim=-1)
    confs, idxs = probs.max(dim=-1)
    results = [{"text": t, "class_index": int(i), "class_name": conf.id2name(int(i)),
                "prob": round(float(p), 4)} for t, i, p in zip(texts, idxs, confs)]
    return (results, model) if return_model else results


if __name__ == "__main__":
    conf = Config()
    model, best_acc, test_loader = train(conf, limit=None)
    test_and_report(conf, model)
    # 防止 CUDA 清理崩溃触发任务框架重跑（同 distill.py 注释）
    os._exit(0)
