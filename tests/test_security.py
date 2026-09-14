# -*- coding: utf-8 -*-
"""服务安全行为测试：鉴权 / 入参上限 / 限流（TestClient 进程内，真走 HTTP 栈）。

需要模型权重的用例（429/200）在本机跑（权重存在时生效）；CI 单测阶段无权重自动跳过。"""
import importlib
import json
import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("DATA_VERSION", "v2")  # 安全测试跟随当前主线 v2（15 类）

import src.serving.app as app_mod
from config.config import Config

_conf = Config()
_WEIGHTS = _conf.student_save_path.replace(".pt", "_soft.pt")
_HAS_WEIGHTS = os.path.exists(_WEIGHTS)

KEY = "test-key-123"


def _reload_with(env: dict) -> None:
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    importlib.reload(app_mod)


@pytest.fixture()
def auth_on(monkeypatch):
    monkeypatch.setenv("API_AUTH_KEY", KEY)
    monkeypatch.setenv("MODEL_DEVICE", "cpu")
    _reload_with({})
    yield
    _reload_with({"API_AUTH_KEY": None})  # 还原为开放模式，避免污染其他用例


@pytest.fixture()
def auth_off(monkeypatch):
    monkeypatch.delenv("API_AUTH_KEY", raising=False)
    monkeypatch.setenv("MODEL_DEVICE", "cpu")
    _reload_with({})
    yield


def test_401_without_key(auth_on):
    client = TestClient(app_mod.app)
    r = client.post("/predict", json={"texts": "寄两份合同文件"})
    assert r.status_code == 401


def test_401_wrong_key(auth_on):
    client = TestClient(app_mod.app)
    r = client.post("/predict", json={"texts": "寄两份合同文件"}, headers={"X-API-Key": "wrong"})
    assert r.status_code == 401


def test_400_empty_texts(auth_off):
    client = TestClient(app_mod.app)
    assert client.post("/predict", json={"texts": ""}).status_code == 400
    assert client.post("/predict", json={"texts": ["  "]}).status_code == 400


def test_400_oversize_batch_and_length(auth_off):
    client = TestClient(app_mod.app)
    r = client.post("/predict", json={"texts": ["ok"] * (app_mod.MAX_BATCH + 1)})
    assert r.status_code == 400
    r = client.post("/predict", json={"texts": ["x" * (app_mod.MAX_TEXT_CHARS + 1)]})
    assert r.status_code == 400


@pytest.mark.skipif(not _HAS_WEIGHTS, reason="需要蒸馏学生权重（本机全量回归跑，CI 单测阶段跳过）")
def test_200_with_valid_key(auth_on):
    client = TestClient(app_mod.app)
    r = client.post("/predict", json={"texts": ["寄两份合同文件", "一箱东西"]},
                    headers={"X-API-Key": KEY})
    assert r.status_code == 200, r.text
    assert len(r.json()) == 2
    assert all("needs_human_review" in item for item in r.json())


@pytest.mark.skipif(not _HAS_WEIGHTS, reason="需要蒸馏学生权重（本机全量回归跑，CI 单测阶段跳过）")
def test_429_rate_limit(auth_off):
    client = TestClient(app_mod.app)
    codes = [client.post("/predict", json={"texts": "一箱东西"}).status_code
             for _ in range(app_mod.RATE_LIMIT + 1)]
    assert all(c == 200 for c in codes[:app_mod.RATE_LIMIT])
    assert codes[-1] == 429


def test_health_open_even_with_auth(auth_on):
    """/health 不鉴权，保证探活可用。"""
    client = TestClient(app_mod.app)
    assert client.get("/health").status_code == 200


# ---------- 护栏行为（红线词库 / OOV / feedback 幂等与上限） ----------

def _write_words(tmp_path, words):
    p = tmp_path / "words.json"
    p.write_text(json.dumps(words), encoding="utf-8")
    return os.fspath(p)


def test_redline_hit_matches(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod, "_REDLINE_WORDS_PATH", _write_words(tmp_path, ["小猫", "充电宝"]))
    monkeypatch.setattr(app_mod, "_redline_words", None)
    assert "小猫" in app_mod._redline_hit("寄小猫一只")
    assert app_mod._redline_hit("一箱车厘子") == []


def test_oov_unseen_marks_novel(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod, "_OOV_VOCAB_PATH", _write_words(tmp_path, ["车厘子", "手表"]))
    monkeypatch.setattr(app_mod, "_oov_vocab", None)
    assert "炒锅" in app_mod._unseen_words("寄一口炒锅")
    assert app_mod._unseen_words("车厘子") == []


def test_feedback_auth_and_dedup(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod, "FEEDBACK_AUTH_KEY", "fb-key-9")
    monkeypatch.setattr(app_mod, "FEEDBACK_DIR", os.fspath(tmp_path))
    monkeypatch.setattr(app_mod, "_fb_seen", None)
    client = TestClient(app_mod.app)
    body = dict(sn="fb1", text="小猫", human_class=13, model_pred=4,
                model_conf=0.93, model_ver="t", ts="t")
    assert client.post("/feedback", json=body).status_code == 401
    hd = {"X-API-Key": "fb-key-9"}
    r = client.post("/feedback", json=body, headers=hd)
    assert r.status_code == 200 and r.json()["accepted"] is True
    assert client.post("/feedback", json=body, headers=hd).json()["dedup"] is True


def test_feedback_max_sn_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(app_mod, "FEEDBACK_AUTH_KEY", "fb-key-9")
    monkeypatch.setattr(app_mod, "FEEDBACK_MAX_SN", 2)
    monkeypatch.setattr(app_mod, "FEEDBACK_DIR", os.fspath(tmp_path))
    monkeypatch.setattr(app_mod, "_fb_seen", None)
    client = TestClient(app_mod.app)
    hd = {"X-API-Key": "fb-key-9"}
    for i in range(2):
        r = client.post("/feedback", json=dict(sn=f"s{i}", text=f"物品{i}", human_class=3,
                                               model_pred=3, model_conf=0.9, model_ver="t", ts="t"),
                        headers=hd)
        assert r.status_code == 200
    r = client.post("/feedback", json=dict(sn="s9", text="物品9", human_class=3, model_pred=3,
                                           model_conf=0.9, model_ver="t", ts="t"), headers=hd)
    assert r.status_code == 429
