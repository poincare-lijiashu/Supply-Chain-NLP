# -*- coding: utf-8 -*-
"""学生模型：BiLSTM（embed=128, hidden=256, 2层双向），参数量约为教师的 1/18。"""
import torch
import torch.nn as nn

from config.config import Config


class BiLSTMClassifier(nn.Module):
    def __init__(self, conf: Config):
        super().__init__()
        vocab_size = conf.bert_config.vocab_size if hasattr(conf, "bert_config") else 21128
        self.embedding = nn.Embedding(vocab_size, conf.student_embed)
        self.lstm = nn.LSTM(conf.student_embed, conf.student_hidden, conf.student_layers,
                            bidirectional=True, batch_first=True, dropout=conf.student_dropout)
        self.hidden_projection = nn.Linear(conf.student_hidden * 2, conf.hidden_size)  # 对齐768维
        self.fc = nn.Linear(conf.student_hidden * 2, conf.num_classes)
        self.dropout = nn.Dropout(conf.student_dropout)

    def forward(self, input_ids, attention_mask, return_hidden=False):
        embed = self.embedding(input_ids) * attention_mask.unsqueeze(-1)
        lstm_out, _ = self.lstm(embed)
        hidden = lstm_out[:, -1, :]
        hidden = self.dropout(hidden)
        logits = self.fc(hidden)
        if return_hidden:
            return logits, self.hidden_projection(hidden)
        return logits


def param_size_mb(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
