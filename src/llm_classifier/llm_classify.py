# -*- coding: utf-8 -*-
"""LLM 分类对照实验：DeepSeek API + few-shot 提示词 + JSON 输出 + 重试。
小样本评估：验证「LLM 准确率低于微调 BERT 且单条成本/延迟高」的选型论证。"""
import json
import os
import sys
import time

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed

from config.config import Config

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

CLASSES = ["文件资料", "服饰鞋帽", "数码家电", "食品生鲜", "日用百货",
           "美妆个护", "医药健康", "易碎品", "违禁品", "其他"]

PROMPT = f"""你是一名快递寄件初审审核员，任务是将用户填写的托寄物描述/寄件备注分类到以下 10 个类目之一：
{json.dumps(CLASSES, ensure_ascii=False)}

类目说明与示例：
- 文件资料：合同、标书、发票、证件、图纸（例："寄两份合同文件"）
- 服饰鞋帽：衣服、鞋、帽、配饰（例："两双运动鞋"）
- 数码家电：手机、电脑、家电（例："一部华为手机"）
- 食品生鲜：水果、海鲜、零食、冷冻食品（例："一箱车厘子"）
- 日用百货：洗护纸品、家居小件（例："洗衣液2瓶"）
- 美妆个护：化妆品、护肤、香氛（例："精华液3瓶"）
- 医药健康：奶粉、保健品、健康器械（例："给小孩寄的奶粉"）
- 易碎品：陶瓷、玻璃、易碎工艺品（例："一箱车厘子玻璃瓶装"？不，应是"陶瓷碗4个"）
- 违禁品：充电宝、打火机、酒精、烟花、管制刀具等寄递红线物品（例："充电宝2个"）
- 其他：无法归入以上类目的描述（例："一包杂物"）

注意：液体化妆品若为玻璃瓶装，优先归"美妆个护"；充电宝/锂电池无论是否全新一律归"违禁品"。
只返回 JSON：{{"category": "类目名", "reason": "一句话理由"}}"""


def build_client():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        base_url=os.getenv("BASE_URL", "https://api.deepseek.com/v1"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        model=os.getenv("LLM_MODEL", "deepseek-flash"),
        model_kwargs={"response_format": {"type": "json_object"}},
    )


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def invoke_llm(llm, text: str) -> dict:
    resp = llm.invoke([{"role": "system", "content": PROMPT},
                       {"role": "user", "content": f"托寄物描述：'{text}'，请分类。"}])
    content = resp.content.strip()
    if content.startswith("```"):  # 防御：部分网关会给 JSON 包 markdown 围栏
        content = content.strip("`").removeprefix("json").strip()
    return json.loads(content)


def run_eval(conf: Config, n: int = 51):
    from sklearn.metrics import accuracy_score, classification_report
    rows = []
    with open(conf.dev_path, "r", encoding="utf-8") as f:
        for line in f:
            text, label = line.rstrip("\n").rsplit("\t", 1)
            rows.append((text, int(label)))
            if len(rows) >= n:
                break

    llm = build_client()
    name2idx = {v: k for k, v in enumerate(conf.class_list)}
    preds, truths = [], []
    t0 = time.time()
    for text, label in rows:
        try:
            result = invoke_llm(llm, text)
            cat = result.get("category", "其他")
            pred = name2idx.get(cat, conf.class_list.index("其他"))
        except Exception as e:  # noqa: BLE001
            print(f"  LLM 调用失败: {e}")
            pred = conf.class_list.index("其他")
        preds.append(pred)
        truths.append(label)
        print(f"  {text!r} -> {conf.class_list[pred]} (真值 {conf.class_list[label]})")
    elapsed = time.time() - t0

    acc = accuracy_score(truths, preds)
    report = classification_report(truths, preds, target_names=conf.class_list,
                                   zero_division=0, digits=4)
    summary = (f"DeepSeek LLM 分类对照实验（{n} 条 dev 样本）\n"
               f"Acc={acc:.4f}, 总耗时={elapsed:.0f}s, 平均 {elapsed / n * 1000:.0f} ms/条\n"
               f"（微调 BERT / 蒸馏对照见 experiments/bert.md 与 distill.md；LLM 单条成本与延迟远高于本地模型——选型论证用）\n\n{report}")
    print(summary)
    return summary


if __name__ == "__main__":
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("未配置 .env 的 DEEPSEEK_API_KEY，跳过 LLM 实验（复制 .env.example 填入后重试）")
        sys.exit(0)
    out = os.path.join(PROJECT_ROOT, "experiments", "llm_classify.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("# LLM 分类对照实验\n\n```\n" + run_eval(Config()) + "\n```\n")
    print(f"已保存 -> {out}")
