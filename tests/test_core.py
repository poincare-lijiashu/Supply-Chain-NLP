# -*- coding: utf-8 -*-
"""核心算法单测：蒸馏损失公式 / 人工复核判定 / 入参校验 / 配置 / 学生模型前向。
不依赖模型权重与数据文件（Config 仅读 class.txt），CPU 秒级完成。"""
import torch
import torch.nn as nn
import pytest

from config.config import Config
from src.compress.bilstm import BiLSTMClassifier
from src.compress.distill import soft_distill_loss
from src.serving.app import MAX_BATCH, MAX_TEXT_CHARS, review_decision, validate_texts


# ---------- 蒸馏损失（KL·T² + α 加权） ----------

def test_soft_distill_loss_zero_when_teacher_student_aligned():
    """教师与学生 logits 完全一致且 alpha=1：软损失为 0（KL 恒为 0）。"""
    t = torch.tensor([[2.0, -1.0, 0.5], [0.1, 3.0, -2.0]])
    y = torch.tensor([0, 1])
    assert soft_distill_loss(t.clone(), t, y, T=2.0, alpha=1.0).item() == pytest.approx(0.0, abs=1e-6)


def test_soft_distill_loss_alpha_zero_equals_ce():
    """alpha=0 退化为纯交叉熵（蒸馏公式加权正确性）。"""
    torch.manual_seed(0)
    s, t = torch.randn(4, 10), torch.randn(4, 10)
    y = torch.randint(0, 10, (4,))
    ce = nn.CrossEntropyLoss()
    got = soft_distill_loss(s, t, y, T=2.0, alpha=0.0, ce=ce).item()
    assert got == pytest.approx(ce(s, y).item(), rel=1e-6)


def test_soft_distill_loss_alpha_one_pure_kl_scaled_by_T2():
    """alpha=1 且 T=1：损失即标准 KL(student‖teacher)。"""
    torch.manual_seed(1)
    s, t = torch.randn(4, 10), torch.randn(4, 10)
    y = torch.randint(0, 10, (4,))
    kl = torch.nn.functional.kl_div(
        torch.log_softmax(s, dim=1), torch.softmax(t, dim=1), reduction="batchmean")
    got = soft_distill_loss(s, t, y, T=1.0, alpha=1.0).item()
    assert got == pytest.approx(kl.item(), rel=1e-6)


def test_temperature_softens_teacher_distribution():
    """温度升高使教师分布变软（T 的作用方向正确）。"""
    t = torch.tensor([[8.0] + [0.0] * 9])
    p1 = torch.softmax(t / 1.0, dim=-1).max().item()
    p4 = torch.softmax(t / 4.0, dim=-1).max().item()
    assert p4 < p1


# ---------- 人工复核判定（低置信度 / 违禁品红线） ----------

def test_review_decision_low_confidence():
    needs, reason = review_decision(0.40, "文件资料", 0.60)
    assert needs and "0.6" in reason and "0.40" in reason


def test_review_decision_prohibited_redline():
    """高置信度违禁品仍必须转人工（红线策略）。"""
    needs, reason = review_decision(0.99, "违禁品", 0.60)
    assert needs and "违禁品" in reason


def test_review_decision_normal_pass():
    needs, reason = review_decision(0.85, "文件资料", 0.60)
    assert not needs and reason == ""


# ---------- 入参校验（条数 / 长度 / 非空） ----------

def test_validate_texts_normalizes_str_to_list():
    assert validate_texts("寄两份合同文件") == ["寄两份合同文件"]
    assert validate_texts(["a", "b"]) == ["a", "b"]


def test_validate_texts_rejects_empty_or_blank():
    for bad in ["", "   ", [], ["  "], ["ok", ""]]:
        with pytest.raises(ValueError):
            validate_texts(bad)


def test_validate_texts_rejects_oversize_batch_and_length():
    with pytest.raises(ValueError):
        validate_texts(["x" * (MAX_TEXT_CHARS + 1)])
    with pytest.raises(ValueError):
        validate_texts(["ok"] * (MAX_BATCH + 1))


# ---------- 配置 ----------

def test_config_id2name_bounds():
    conf = Config()
    assert conf.id2name(0) == "文件资料"
    assert conf.id2name(conf.num_classes - 1) == conf.class_list[-1]
    assert conf.id2name(99) == "其他"
    assert conf.id2name(-1) == "其他"


def test_config_security_defaults():
    conf = Config()
    assert conf.serve_host == "127.0.0.1"          # 默认仅本机
    assert 0 < conf.api_max_batch <= 1000
    assert 0 < conf.api_max_text_chars <= 4096
    assert 0 < conf.api_rate_limit_per_min <= 10000
    assert 0 < conf.confidence_threshold < 1


# ---------- 学生模型前向（小尺寸，不加载权重） ----------

def test_bilstm_forward_shapes():
    conf = Config()
    conf.student_embed, conf.student_hidden = 8, 16
    conf.student_layers, conf.student_dropout = 1, 0.0
    model = BiLSTMClassifier(conf)
    ids = torch.randint(0, 21128, (2, 5))
    mask = torch.ones(2, 5, dtype=torch.long)
    logits = model(ids, mask)
    assert logits.shape == (2, conf.num_classes)
    _, hidden = model(ids, mask, return_hidden=True)
    assert hidden.shape == (2, conf.hidden_size)   # 中间层对齐教师 768 维
