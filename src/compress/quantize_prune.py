# -*- coding: utf-8 -*-
"""模型量化（动态量化 int8，仅 CPU）与剪枝（L1 非结构化 30%）+ 体积/延迟基准。"""
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
import torch  # noqa: E402
import torch.nn.utils.prune as prune  # noqa: E402
from config.config import Config  # noqa: E402
from src.bert_model.bert_pipeline import BertClassifier, build_loaders, evaluate  # noqa: E402


def file_size_mb(path: str) -> float:
    return os.path.getsize(path) / 1024 / 1024


def quantize(conf: Config):
    """动态量化：Linear 权重 float32 -> int8，仅 CPU 执行。"""
    _, _, test_loader = build_loaders(conf)
    model = BertClassifier(conf)  # CPU
    model.load_state_dict(torch.load(conf.model_save_path, map_location="cpu"))
    model.eval()

    q_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    acc, f1, report, cm, _, _ = evaluate(q_model, test_loader, conf, device=torch.device("cpu"))
    q_path = conf.model_save_path.replace(".pt", "_int8.pt")
    torch.save(q_model, q_path)
    summary = (f"动态量化(int8, Linear) Test Acc={acc:.4f}, Macro F1={f1:.4f}\n"
               f"模型体积: fp32 {file_size_mb(conf.model_save_path):.1f}MB -> int8 {file_size_mb(q_path):.1f}MB\n"
               f"量化模型: {q_path}\n{report}")
    print(summary)
    return summary, q_path, acc


def prune_model(conf: Config):
    """对 12 层 encoder 的 attention query 权重做 L1 全局非结构化剪枝 30%。"""
    _, _, test_loader = build_loaders(conf)
    model = BertClassifier(conf).to(conf.device)
    model.load_state_dict(torch.load(conf.model_save_path, map_location=conf.device))
    acc0, f10, _, _, _, _ = evaluate(model, test_loader, conf, full=False)

    params_to_prune = [(model.bert.encoder.layer[i].attention.self.query, "weight") for i in range(12)]
    prune.global_unstructured(params_to_prune, pruning_method=prune.L1Unstructured,
                              amount=conf.prune_amount)
    for module, param in params_to_prune:
        prune.remove(module, param)

    total, zeros = 0, 0
    for i in range(12):
        w = model.bert.encoder.layer[i].attention.self.query.weight
        total += w.numel()
        zeros += (w == 0).sum().item()
    sparsity = zeros / total
    acc1, f11, _, _, _, _ = evaluate(model, test_loader, conf, full=False)
    p_path = conf.model_save_path.replace(".pt", "_pruned.pt")
    torch.save(model.state_dict(), p_path)
    summary = (f"L1剪枝30%（encoder.query）: Acc {acc0:.4f}->{acc1:.4f}, F1 {f10:.4f}->{f11:.4f}, "
               f"稀疏度={sparsity:.4f}\n剪枝模型: {p_path}")
    print(summary)
    return summary


def benchmark(conf: Config, n: int = 100):
    """单条推理延迟对比：BERT fp32 vs int8（CPU 口径，与部署场景一致）。"""
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(conf.pretrain_bert_dir)
    sample = "寄两份合同文件明天必须到"

    def latency(model, device):
        enc = tokenizer([sample], add_special_tokens=True, padding=True, truncation=True,
                        max_length=conf.max_len, return_attention_mask=True, return_tensors="pt")
        ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        model.eval()
        with torch.no_grad():
            model(ids, mask)  # 预热
            t0 = time.time()
            for _ in range(n):
                model(ids, mask)
        return (time.time() - t0) / n * 1000

    fp32_model = BertClassifier(conf)  # CPU
    fp32_model.load_state_dict(torch.load(conf.model_save_path, map_location="cpu"))
    q_model = torch.quantization.quantize_dynamic(fp32_model, {torch.nn.Linear}, dtype=torch.qint8)
    cpu = torch.device("cpu")
    lat_fp32 = latency(fp32_model, cpu)
    lat_int8 = latency(q_model, cpu)
    lines = [f"BERT fp32 (CPU): {lat_fp32:.1f} ms/条",
             f"BERT int8 (CPU): {lat_int8:.1f} ms/条",
             f"加速比: {lat_fp32 / lat_int8:.2f}x"]
    print("\n".join(lines))
    return "\n".join(lines)


if __name__ == "__main__":
    conf = Config()
    results = {}
    results["quantize"], q_path, q_acc = quantize(conf)
    results["prune"] = prune_model(conf)
    results["benchmark"] = benchmark(conf)
    out = os.path.join(PROJECT_ROOT, "experiments", "compress.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("# 模型压缩实验（量化 / 剪枝 / 延迟基准）\n\n```\n" +
                "\n\n".join(results.values()) + "\n```\n")
    print(f"已保存 -> {out}")
