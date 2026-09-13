# -*- coding: utf-8 -*-
"""冒烟测试：小样本跑通「数据 -> RF -> fasttext -> BERT -> 蒸馏 -> 量化 -> 服务推理」全链路。
用法：python -m scripts.run_smoke   （项目根目录执行；每模块 2000 条训练，几分钟内完成，CPU 也能跑）"""
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from config.config import Config  # noqa: E402

conf = Config()
SMOKE = 2000
SMOKE_BATCH = int(os.getenv("SMOKE_BATCH", "128"))  # 内存紧张时可 SMOKE_BATCH=32 降低峰值占用
conf.batch_size = SMOKE_BATCH
SMOKE_DIR = os.path.join(PROJECT_ROOT, "data", "smoke")
ok = []


def step(name, fn):
    print(f"\n===== [冒烟] {name} =====")
    fn()
    ok.append(name)


def _gen():
    r = subprocess.run([sys.executable, "-m", "data.generate_data",
                        "--train", str(SMOKE), "--dev", "400", "--test", "400",
                        "--outdir", SMOKE_DIR], check=False, cwd=PROJECT_ROOT)
    if r.returncode != 0:
        raise RuntimeError("generate fail")


step("数据生成（小样本, data/smoke）", _gen)

# 冒烟全程使用 data/smoke 数据，绝不覆盖 data/raw 全量数据
conf.train_path = os.path.join(SMOKE_DIR, "train.txt")
conf.dev_path = os.path.join(SMOKE_DIR, "dev.txt")
conf.test_path = os.path.join(SMOKE_DIR, "test.txt")
conf.process_dir = os.path.join(SMOKE_DIR, "processed")
conf.process_train_path = os.path.join(SMOKE_DIR, "processed", "train_process.csv")
conf.process_dev_path = os.path.join(SMOKE_DIR, "processed", "dev_process.csv")
conf.process_test_path = os.path.join(SMOKE_DIR, "processed", "test_process.csv")
conf.fasttext_dir = os.path.join(SMOKE_DIR, "fasttext")
conf.train_ft_char_path = os.path.join(SMOKE_DIR, "fasttext", "train_ft_char.txt")
conf.dev_ft_char_path = os.path.join(SMOKE_DIR, "fasttext", "dev_ft_char.txt")
conf.test_ft_char_path = os.path.join(SMOKE_DIR, "fasttext", "test_ft_char.txt")
conf.train_ft_jieba_path = os.path.join(SMOKE_DIR, "fasttext", "train_ft_jieba.txt")
conf.dev_ft_jieba_path = os.path.join(SMOKE_DIR, "fasttext", "dev_ft_jieba.txt")
conf.test_ft_jieba_path = os.path.join(SMOKE_DIR, "fasttext", "test_ft_jieba.txt")
# 模型产物也全部隔离到 models/smoke/，绝不覆盖正式权重
conf.rf_model_dir = os.path.join(PROJECT_ROOT, "models", "smoke", "baseline_rf")
conf.ft_model_dir = os.path.join(PROJECT_ROOT, "models", "smoke", "fasttext")
conf.model_save_dir = os.path.join(PROJECT_ROOT, "models", "smoke", "checkpoints")
conf.model_save_path = os.path.join(PROJECT_ROOT, "models", "smoke", "checkpoints", "bert_best.pt")
conf.student_save_path = os.path.join(PROJECT_ROOT, "models", "smoke", "checkpoints", "distill_best.pt")
for d in [conf.rf_model_dir, conf.ft_model_dir, conf.model_save_dir]:
    os.makedirs(d, exist_ok=True)

from src.baseline_rf.rf_baseline import train_and_eval  # noqa: E402


def rf_smoke():
    for f in [conf.process_train_path, conf.process_dev_path]:
        if os.path.exists(f):
            os.remove(f)
    train_and_eval(conf, subset=SMOKE)
    for f in [conf.process_train_path, conf.process_dev_path]:
        if os.path.exists(f):
            os.remove(f)


step("RF 基线", rf_smoke)

from src.bert_model.bert_pipeline import train as bert_train, test_and_report  # noqa: E402


def bert_smoke():
    model, best_acc, _ = bert_train(conf, limit=SMOKE)
    test_and_report(conf, model)


step("BERT 微调（小样本）", bert_smoke)

from src.compress.distill import distill  # noqa: E402
import transformers  # noqa: E402


def distill_smoke():
    from transformers import BertConfig
    conf.bert_config = BertConfig.from_pretrained(conf.pretrain_bert_dir)
    distill(conf, mode="soft", limit=SMOKE)


step("知识蒸馏（软标签, 小样本）", distill_smoke)

from src.compress.quantize_prune import quantize  # noqa: E402


def quant_smoke():
    quantize(conf)


step("动态量化", quant_smoke)

def _serve_smoke():
    from fastapi.testclient import TestClient
    os.environ["MODEL_NAME"] = "distill"
    import importlib
    import src.serving.app as app_mod
    importlib.reload(app_mod)
    # reload 会重建默认路径的 Config（指向 models/checkpoints），此处回填冒烟隔离路径，
    # 保证 CI（无正式权重）也全程使用 models/smoke 产物
    app_mod.conf.model_save_path = conf.model_save_path
    app_mod.conf.student_save_path = conf.student_save_path
    client = TestClient(app_mod.app)
    resp = client.post("/predict", json={"texts": ["寄两份合同文件", "一箱东西"]})
    assert resp.status_code == 200, resp.text
    results = resp.json()
    print(results)
    assert all("needs_human_review" in r for r in results)


step("FastAPI 推理函数（进程内）", _serve_smoke)

print(f"\n冒烟通过：{len(ok)}/{len(ok)} -> {ok}")
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)  # 规避 Windows 上 torch/tqdm 解释器关闭阶段的原生崩溃（0xC0000409，不影响产出）
