# -*- coding: utf-8 -*-
"""FastAPI 部署（简历口径）：/predict 单条+批量，低置信度转人工复核（违禁品红线策略）。

启动：uvicorn src.serving.app:app --host 0.0.0.0 --port 8004
模型选择：MODEL_NAME=distill|bert_int8|bert 环境变量，默认蒸馏模型（上线口径）。
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
import torch  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from transformers import BertTokenizer  # noqa: E402
from config.config import Config  # noqa: E402
from src.bert_model.bert_pipeline import BertClassifier  # noqa: E402
from src.compress.bilstm import BiLSTMClassifier  # noqa: E402

conf = Config()
app = FastAPI(title="物流寄件文本智能分类服务", version="1.1.0")


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
                                         map_location=conf.device))
    elif choice == "bert_int8":
        path = conf.model_save_path.replace(".pt", "_int8.pt")
        model = torch.quantization.quantize_dynamic(
            BertClassifier(conf), {torch.nn.Linear}, dtype=torch.qint8)
        model.load_state_dict(torch.load(path, map_location="cpu"))
    else:
        model = BertClassifier(conf).to(conf.device)
        model.load_state_dict(torch.load(conf.model_save_path, map_location=conf.device))
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(conf.pretrain_bert_dir)
    return model, tokenizer


_model, _tokenizer = None, None


def _get():
    global _model, _tokenizer
    if _model is None:
        _model, _tokenizer = _load_model()
    return _model, _tokenizer


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
        needs_review, reason = False, ""
        if p < conf.confidence_threshold:
            needs_review, reason = True, f"置信度{p:.2f}低于阈值{conf.confidence_threshold}，转人工复核"
        elif conf.class_list[idx] == "违禁品":
            needs_review, reason = True, "违禁品类目强制人工复核（寄递安全红线）"
        items.append(PredictItem(text=text, class_index=idx, class_name=conf.class_list[idx],
                                 prob=round(float(p), 4), probs=prob_map,
                                 needs_human_review=needs_review, review_reason=reason))
    return items


@app.post("/predict", response_model=list[PredictItem])
def predict(req: PredictRequest):
    texts = [req.texts] if isinstance(req.texts, str) else req.texts
    if not texts or not all(isinstance(t, str) and t.strip() for t in texts):
        raise HTTPException(status_code=400, detail="texts 必须为非空字符串或字符串列表")
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
