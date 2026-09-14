# -*- coding: utf-8 -*-
"""学生模型：BiLSTM（embed=128, hidden=256, 2层双向），参数量约为教师的 1/18。

支持 BERT 字向量初始化（conf.init_from_bert）：
  用 BERT 的 embedding 矩阵（768维）经随机投影到 student_embed 维，作为学生 embedding 初值，
  让 21MB 小模型继承 BERT 的中文字义常识（对训练集 0 出现的裸词（西瓜/轴承）有泛化）。
  对照：随机初始化（torch默认）保留，供 P12 双轨对比。
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from config.config import Config


class BiLSTMClassifier(nn.Module):
    def __init__(self, conf: Config):
        super().__init__()
        vocab_size = conf.bert_config.vocab_size if hasattr(conf, "bert_config") else 21128
        self.conf = conf
        self.embedding = nn.Embedding(vocab_size, conf.student_embed)
        # BERT 字向量初始化（若启用）：从预训练 BERT embedding 投影到学生维度
        if getattr(conf, "init_from_bert", False):
            self._init_embed_from_bert()
        self.lstm = nn.LSTM(conf.student_embed, conf.student_hidden, conf.student_layers,
                            bidirectional=True, batch_first=True, dropout=conf.student_dropout)
        self.hidden_projection = nn.Linear(conf.student_hidden * 2, conf.hidden_size)  # 对齐768维
        self.fc = nn.Linear(conf.student_hidden * 2, conf.num_classes)
        self.dropout = nn.Dropout(conf.student_dropout)

    def _init_embed_from_bert(self):
        """用 BERT embedding 软初始化学生字向量（投影 + 保持随机噪声扰动小，防直接退化）。"""
        from transformers import BertModel
        bert = BertModel.from_pretrained(self.conf.pretrain_bert_dir, output_hidden_states=False)
        # 取 word_embeddings（[vocab_size, 768]），与学生共享 vocab_size 词表
        b_emb = bert.embeddings.word_embeddings.weight.data  # (V, 768)
        proj = torch.randn(768, self.conf.student_embed) * 0.02  # 随机投影，保持各向异性
        init = b_emb.cpu() @ proj  # (V, student_embed)
        # freeze BERT 权重（仅用于初始化，不参与学生训练）
        del bert
        # 学生 embedding 用投影结果 + 微噪（0.01 尺度）
        self.embedding.weight.data.copy_(init + torch.randn_like(init) * 0.01)
        # embedding 保持可训练（微调词向量），不冻结

    def forward(self, input_ids, attention_mask, return_hidden=False):
        embed = self.embedding(input_ids)
        # pack 让双向 LSTM 物理跳过 padding：正反向均只沿真实 token 递推，
        # 否则反向 LSTM 会先扫过尾部零向量填充位，导致单条（无 padding）与批量
        # 请求的句向量不同——训练/测试大批 padding、上线单条无 padding 的分布漂移。
        lengths = attention_mask.sum(dim=1).long()
        packed = pack_padded_sequence(embed, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)  # h_n: (num_layers*2, B, H)，已还原原始顺序
        # 最后一层：正向「最后真实 token」状态 + 反向「第一真实 token」状态
        hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        hidden = self.dropout(hidden)
        logits = self.fc(hidden)
        if return_hidden:
            return logits, self.hidden_projection(hidden)
        return logits


def param_size_mb(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
