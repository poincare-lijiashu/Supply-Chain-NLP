# -*- coding: utf-8 -*-
"""部署验证客户端：单条 + 批量预测，验证置信度阈值转人工复核逻辑。"""
import json
import sys
import urllib.request

URL = "http://127.0.0.1:8004/predict"


def predict(texts):
    req = urllib.request.Request(
        URL, data=json.dumps({"texts": texts}).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


if __name__ == "__main__":
    cases = [
        "寄两份合同文件，明天必须到",
        "一箱车厘子",
        "充电宝2个",
        "给小孩寄的奶粉",
        "一箱东西",              # 歧义样本 -> 大概率低置信度转人工
    ]
    results = predict(cases)
    for r in results:
        flag = "  [转人工] " + r["review_reason"] if r["needs_human_review"] else ""
        print(f"{r['text']!r:32} -> {r['class_name']}({r['prob']:.2f}){flag}")
