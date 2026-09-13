# -*- coding: utf-8 -*-
"""服务安全行为测试：鉴权 / 入参上限 / 限流（TestClient 进程内，真走 HTTP 栈）。

需要模型权重的用例（429/200）在本机跑（权重存在时生效）；CI 单测阶段无权重自动跳过。"""
import importlib
import os

import pytest
from fastapi.testclient import TestClient

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
