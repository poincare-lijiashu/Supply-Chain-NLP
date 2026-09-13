# -*- coding: utf-8 -*-
"""FastAPI 推理服务：/predict 单条+批量，低置信度转人工复核（违禁品红线策略）。

启动：uvicorn src.serving.app:app --host 127.0.0.1 --port 8004
模型选择：MODEL_NAME=distill|bert_int8|bert 环境变量，默认蒸馏模型（上线口径）。
鉴权：设置环境变量 API_AUTH_KEY 后，/predict 要求请求头 X-API-Key 精确匹配（/health 不鉴权）；
      不设置则为本机开放模式——仅供 127.0.0.1 本机调试，切勿直接暴露公网。
限流：内存滑动窗口，每客户端每分钟最多 API_RATE_LIMIT_PER_MIN 次请求（单进程 uvicorn 口径）。
"""
import hmac
import os
import threading
import time
from collections import defaultdict, deque

import torch
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import BertTokenizer

from config.config import Config
from src.bert_model.bert_pipeline import BertClassifier
from src.compress.bilstm import BiLSTMClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

conf = Config()
app = FastAPI(title="物流寄件文本智能分类服务", version="1.2.0")

AUTH_KEY = os.getenv("API_AUTH_KEY", "").strip()
MAX_BATCH = conf.api_max_batch              # 单次请求最大条数
MAX_TEXT_CHARS = conf.api_max_text_chars    # 单条文本最大字符数
RATE_LIMIT = conf.api_rate_limit_per_min    # 每客户端每分钟最大请求数


class PredictRequest(BaseModel):
    texts: str | list[str] = Field(..., description="托寄物描述，字符串或字符串列表")


class PredictItem(BaseModel):
    text: str
    class_index: int
    class_name: str
    prob: float
    probs: dict[str, float] = Field(default_factory=dict, description="全部类目概率")
    needs_human_review: bool
    review_reason: str


def _load_model():
    choice = os.getenv("MODEL_NAME", "distill")
    if choice == "distill":
        model = BiLSTMClassifier(conf).to(conf.device)
        model.load_state_dict(torch.load(conf.student_save_path.replace(".pt", "_soft.pt"),
                                         map_location=conf.device, weights_only=True))
    elif choice == "bert_int8":
        path = conf.model_save_path.replace(".pt", "_int8.pt")
        model = torch.quantization.quantize_dynamic(
            BertClassifier(conf), {torch.nn.Linear}, dtype=torch.qint8)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    else:
        model = BertClassifier(conf).to(conf.device)
        model.load_state_dict(torch.load(conf.model_save_path, map_location=conf.device,
                                         weights_only=True))
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(conf.pretrain_bert_dir)
    return model, tokenizer


_model, _tokenizer = None, None


def _get():
    global _model, _tokenizer
    if _model is None:
        _model, _tokenizer = _load_model()
    return _model, _tokenizer


def review_decision(prob: float, class_name: str, threshold: float) -> tuple[bool, str]:
    """人工复核判定（纯函数，便于单测）：低置信度或违禁品红线 -> 转人工。"""
    if prob < threshold:
        return True, f"置信度{prob:.2f}低于阈值{threshold}，转人工复核"
    if class_name == "违禁品":
        return True, "违禁品类目强制人工复核（寄递安全红线）"
    return False, ""


def validate_texts(texts: str | list[str]) -> list[str]:
    """入参校验（纯函数，便于单测）：非空、条数与单条长度上限，返回规范化列表。"""
    items = [texts] if isinstance(texts, str) else texts
    if not items or not all(isinstance(t, str) and t.strip() for t in items):
        raise ValueError("texts 必须为非空字符串或非空字符串列表")
    if len(items) > MAX_BATCH:
        raise ValueError(f"单次最多 {MAX_BATCH} 条")
    if any(len(t) > MAX_TEXT_CHARS for t in items):
        raise ValueError(f"单条文本最长 {MAX_TEXT_CHARS} 字符")
    return items


@torch.no_grad()
def _infer(texts: list[str]) -> list[PredictItem]:
    model, tokenizer = _get()
    enc = tokenizer(texts, add_special_tokens=True, padding=True, truncation=True,
                    max_length=conf.max_len, return_attention_mask=True, return_tensors="pt")
    logits = model(enc["input_ids"].to(conf.device), enc["attention_mask"].to(conf.device))
    probs = torch.softmax(logits, dim=-1)
    confs, idxs = probs.max(dim=-1)
    items = []
    for i, (text, idx, p) in enumerate(zip(texts, idxs.tolist(), confs.tolist())):
        prob_map = {conf.class_list[j]: round(float(probs[i][j]), 4) for j in range(conf.num_classes)}
        needs_review, reason = review_decision(float(p), conf.class_list[idx], conf.confidence_threshold)
        items.append(PredictItem(text=text, class_index=idx, class_name=conf.class_list[idx],
                                 prob=round(float(p), 4), probs=prob_map,
                                 needs_human_review=needs_review, review_reason=reason))
    return items


# ---------- 鉴权与限流（/health 不做限制，保证探活可用） ----------
_rate_buckets: dict[str, deque] = defaultdict(deque)
_rate_lock = threading.Lock()


def _check_auth(api_key: str | None) -> None:
    if not AUTH_KEY:  # 未配置密钥 = 本机开放模式
        return
    if not api_key or not hmac.compare_digest(api_key, AUTH_KEY):
        raise HTTPException(status_code=401, detail="缺少或错误的 X-API-Key 请求头")


def _check_rate(request: Request) -> None:
    client = request.client.host if request.client else "unknown"
    now = time.time()
    with _rate_lock:
        bucket = _rate_buckets[client]
        while bucket and now - bucket[0] > 60:
            bucket.popleft()
        if len(bucket) >= RATE_LIMIT:
            raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")
        bucket.append(now)


@app.post("/predict", response_model=list[PredictItem])
def predict(req: PredictRequest, request: Request, x_api_key: str | None = Header(default=None)):
    _check_auth(x_api_key)
    _check_rate(request)
    try:
        texts = validate_texts(req.texts)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _infer(texts)


@app.get("/health")
def health():
    return {"status": "ok", "model": os.getenv("MODEL_NAME", "distill"),
            "classes": conf.class_list}


# ---------- 前端工作台（无外部 CDN 依赖，内网可用） ----------
_WEB_DIR = os.path.join(PROJECT_ROOT, "web")
if os.path.isdir(_WEB_DIR):
    app.mount("/static", StaticFiles(directory=_WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(os.path.join(_WEB_DIR, "index.html"))
