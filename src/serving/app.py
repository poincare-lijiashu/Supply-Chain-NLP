# -*- coding: utf-8 -*-
"""FastAPI 推理服务：/predict 单条+批量，低置信度转人工复核（违禁品红线策略）。

启动：uvicorn src.serving.app:app --host 127.0.0.1 --port 8004
模型选择：MODEL_NAME=distill|bert_int8|bert 环境变量，默认蒸馏模型（上线口径）；
      v2 阶段蒸馏学生未产出时自动回退教师并告警。
鉴权：设置环境变量 API_AUTH_KEY 后，/predict 要求请求头 X-API-Key 精确匹配（/health 不鉴权）；
      不设置则为本机开放模式——仅供 127.0.0.1 本机调试，切勿直接暴露公网。
限流：内存滑动窗口，每客户端每分钟最多 API_RATE_LIMIT_PER_MIN 次请求（单进程 uvicorn 口径）。
"""
import hmac
import json
import os
import pickle
import re
import threading
import time
from collections import defaultdict, deque

import torch
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import jieba
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

# ---------- OOV 中间带护栏（未登录词 + 中等置信度 → 转人工） ----------
OOV_GUARD = os.getenv("OOV_GUARD", "1") == "1"   # OOV_GUARD=0 关闭
OOV_LOW = float(os.getenv("OOV_LOW", "0.60"))
OOV_HIGH = float(os.getenv("OOV_HIGH", "0.95"))
_OOV_VOCAB_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "oov_vocab.pkl")
_SKIP_TOKEN = re.compile(r"^[\d\W_]+$")
_ASCII_ONLY = re.compile(r"^[A-Za-z]+$")
_oov_vocab: set | None = None

# ---------- 红线命中护栏（禁寄词库命中 → 强制转人工，与置信度无关） ----------
REDLINE_GUARD = os.getenv("REDLINE_GUARD", "1") == "1"
_REDLINE_WORDS_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "redline_words.pkl")
_redline_words: set | None = None

# ---------- 真实标签回流（快递员复检终判 → 逐条 JSONL，按天分片） ----------
FEEDBACK_DIR = os.path.join(PROJECT_ROOT, "data", "feedback")
FEEDBACK_AUTH_KEY = os.getenv("FEEDBACK_AUTH_KEY", "").strip()
FEEDBACK_RATE_LIMIT = int(os.getenv("FEEDBACK_RATE_LIMIT", "120"))
_fb_lock = threading.Lock()
_fb_seen: set | None = None


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


class FeedbackRequest(BaseModel):
    """快递员复检终判回传（真实标签回流）。最小字段，不含 PII（手机号/地址/金额一律不收）。"""
    sn: str = Field(..., min_length=1, max_length=64, description="脱敏单号/内部ID（幂等去重键）")
    text: str = Field(..., min_length=1, max_length=512, description="托寄物描述原件")
    human_class: int = Field(..., ge=0, description="快递员终判类目")
    model_pred: int = Field(..., ge=0, description="模型初判类目")
    model_conf: float = Field(..., ge=0, le=1, description="模型初判置信度")
    model_ver: str = Field(..., min_length=1, max_length=64, description="模型标识（查错/回归关键）")
    model_review: bool = Field(False, description="初判是否被标记转人工")
    human_changed: bool = Field(False, description="快递员是否改判")
    human_note: str = Field(default="", max_length=200, description="改判原因（可选）")
    channel: str = Field(default="", max_length=32, description="渠道 web/ocr/语音 等（可选）")
    ts: str = Field(default="", max_length=32, description="事件时间 ISO，可选则服务端补）")


def _load_model():
    choice = os.getenv("MODEL_NAME", "distill")
    if choice == "distill":
        student_path = conf.student_save_path.replace(".pt", "_soft.pt")
        if not os.path.exists(student_path):
            # v2 阶段蒸馏学生尚未训练：明确告警并回退教师，避免误启崩溃
            print(f"[serving] 蒸馏学生 {student_path} 缺失，回退加载教师 {conf.model_save_path}")
            model = BertClassifier(conf).to(conf.device)
            model.load_state_dict(torch.load(conf.model_save_path, map_location=conf.device,
                                             weights_only=True))
        else:
            model = BiLSTMClassifier(conf).to(conf.device)
            model.load_state_dict(torch.load(student_path,
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
    if "违禁" in class_name:  # v1 违禁品 / v2 违禁限寄 兼容
        return True, "违禁品类目强制人工复核（寄递安全红线）"
    return False, ""


def _unseen_words(text: str) -> list[str]:
    """返回文本中未在训练语料出现过的词元（护栏用）；词表未就绪时降级为空。"""
    global _oov_vocab
    if _oov_vocab is None:
        try:
            with open(_OOV_VOCAB_PATH, "rb") as fh:
                _oov_vocab = pickle.load(fh)
            print(f"[serving] OOV 词表已加载: {len(_oov_vocab)} 词")
        except Exception as e:
            print(f"[serving] OOV 词表加载失败，护栏降级为全放行: {e}")
            _oov_vocab = set()
    unseen: list[str] = []
    for w in jieba.cut(text):
        w = w.strip()
        if not w or _SKIP_TOKEN.match(w) or _ASCII_ONLY.match(w):
            continue
        if w not in _oov_vocab:
            unseen.append(w)
    return unseen


def _redline_hit(text: str) -> list[str]:
    """返回文本中命中的禁寄词库词（红线护栏，长词优先）；词表未就绪降级为空。"""
    global _redline_words
    if _redline_words is None:
        try:
            with open(_REDLINE_WORDS_PATH, "rb") as fh:
                _redline_words = pickle.load(fh)
            print(f"[serving] 红线词表已加载: {len(_redline_words)} 词")
        except Exception as e:
            print(f"[serving] 红线词表加载失败，红线护栏降级为全放行: {e}")
            _redline_words = set()
    hit: list[str] = []
    for w in sorted(_redline_words, key=len, reverse=True):
        if w in text:
            hit.append(w)
            if len(hit) >= 3:
                break
    return hit


def _fb_seen_sns() -> set:
    """已收 sn 集合（进程内缓存 + 扫描已有 JSONL），用于幂等。"""
    global _fb_seen
    if _fb_seen is None:
        sns: set = set()
        if os.path.isdir(FEEDBACK_DIR):
            for fn in os.listdir(FEEDBACK_DIR):
                if not fn.endswith(".jsonl"):
                    continue
                with open(os.path.join(FEEDBACK_DIR, fn), encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            sns.add(json.loads(line).get("sn", ""))
                        except Exception:
                            pass
        _fb_seen = sns
    return _fb_seen


def _save_feedback(payload: dict) -> None:
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    day = time.strftime("%Y-%m-%d")
    with open(os.path.join(FEEDBACK_DIR, f"{day}.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


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
        if not needs_review and OOV_GUARD and OOV_LOW <= float(p) < OOV_HIGH:
            unseen = _unseen_words(text)
            if unseen:
                needs_review, reason = True, "含未登录词：%s（中间带置信），转人工复核" % "、".join(unseen[:3])
        if not needs_review and REDLINE_GUARD:
            rw = _redline_hit(text)
            if rw:
                needs_review, reason = True, "命中禁寄词库：%s（强制复核）" % "、".join(rw)
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


def _check_rate(request: Request, limit: int = RATE_LIMIT) -> None:
    client = request.client.host if request.client else "unknown"
    now = time.time()
    with _rate_lock:
        bucket = _rate_buckets[client]
        while bucket and now - bucket[0] > 60:
            bucket.popleft()
        if len(bucket) >= limit:
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


@app.post("/feedback")
def feedback(req: FeedbackRequest, request: Request, x_api_key: str | None = Header(default=None)):
    """接收快递员复检终判（真实标签回流）。独立 key（FEEDBACK_AUTH_KEY）或复用 API_AUTH_KEY；独立限流。"""
    key = FEEDBACK_AUTH_KEY or AUTH_KEY
    if key and (not x_api_key or not hmac.compare_digest(x_api_key, key)):
        raise HTTPException(status_code=401, detail="缺少或错误的 X-API-Key 请求头")
    _check_rate(request, FEEDBACK_RATE_LIMIT)
    n = conf.num_classes
    if not (0 <= req.human_class < n):
        raise HTTPException(status_code=400, detail=f"human_class 需在 0-{n - 1}")
    if not (0 <= req.model_pred < n):
        raise HTTPException(status_code=400, detail=f"model_pred 需在 0-{n - 1}")
    payload = req.model_dump()
    payload["ts"] = req.ts or time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with _fb_lock:
        if req.sn in _fb_seen_sns():
            return {"accepted": False, "dedup": True, "reason": "sn 已存在"}
        _fb_seen_sns().add(req.sn)
        _save_feedback(payload)
    return {"accepted": True, "dedup": False,
            "dest": f"data/feedback/{time.strftime('%Y-%m-%d')}.jsonl"}


@app.get("/health")
def health():
    return {"status": "ok", "model": os.getenv("MODEL_NAME", "distill"),
            "classes": conf.class_list, "oov_guard": OOV_GUARD}


# ---------- 前端工作台（无外部 CDN 依赖，内网可用） ----------
_WEB_DIR = os.path.join(PROJECT_ROOT, "web")
if os.path.isdir(_WEB_DIR):
    app.mount("/static", StaticFiles(directory=_WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(os.path.join(_WEB_DIR, "index.html"))
