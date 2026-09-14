# 真实标签回流接口对接文档（`POST /feedback`）

版本：1.0　日期：2026-09-14　所属：物流寄件初审分类系统（Supply Chain NLP）

## 1. 背景与目的

系统对寄件人填写的托寄物描述做 15 类自动分类，快递员在复检环节**确认或修改**模型给出的类目。
本文档描述如何把"快递员复检后的最终类目"回传给我方系统——这些真实终判是模型迭代最宝贵的数据
（真实用户表达 + 业务专家校验），用于：真实分布回归、badcase 驱动补词库、后续模型升级训练。

**关键约定（请务必遵守）**：
- **全量回传**：改判的、未改判的**所有订单都要回传**。只回传"改判单"会造成样本偏置，损害迭代质量。
- **最小字段**：只传托寄物描述与类目，**不要传**手机号、地址、收寄件人姓名、金额等个人敏感信息。

## 2. 接口概览

| 项 | 值 |
|---|---|
| 方法 | `POST` |
| 路径 | `/feedback` |
| 内容类型 | `application/json` |
| 鉴权 | 请求头 `X-API-Key: <key>`（与我方 `API_AUTH_KEY` 一致，由对接方提供） |
| 数据格式 | JSON，单条一条请求（循环调用即可，量不大） |

响应状态码：`200` 成功（含重复拒绝）；`400` 参数不合法；`401` 缺少/错误密钥。

## 3. 请求字段

```json
{
  "sn": "hashed_order_001",
  "text": "小猫",
  "human_class": 13,
  "model_pred": 4,
  "model_conf": 0.9393,
  "model_ver": "v2-distill-soft-e3",
  "model_review": false,
  "human_changed": true,
  "human_note": "活体动物不能寄",
  "channel": "web",
  "ts": "2026-09-14T21:00:00+08:00"
}
```

| 字段 | 必填 | 类型 | 约束 | 说明 |
|---|---|---|---|---|
| `sn` | 是 | string | 1-64 字符 | **脱敏单号/内部ID**（对真实单号做哈希或截断，用于幂等去重；同一个单只回传一条） |
| `text` | 是 | string | 1-512 字符 | 寄件人填写的托寄物描述**原文**（不要清洗改写） |
| `human_class` | 是 | int | 0-14 | **快递员复检终判类目**（见 §5 类目表） |
| `model_pred` | 是 | int | 0-14 | 当时系统的初判类目（界面给快递员看的值） |
| `model_conf` | 是 | float | 0-1 | 初判置信度（界面显示的百分比/100） |
| `model_ver` | 是 | string | 1-64 | 初判模型版本（如 `v2-distill-soft-e3`；用于按版本归因，**务必真实填写**） |
| `model_review` | 否 | bool | — | 初判当时是否被系统标记为"转人工复核" |
| `human_changed` | 否 | bool | — | 快递员是否改判（`true`=改过） |
| `human_note` | 否 | string | ≤200 字符 | 改判原因（选填，强烈建议填，价值最高） |
| `channel` | 否 | string | ≤32 | 渠道标识（web/ocr/语音…） |
| `ts` | 否 | string | ISO 时间 | 事件时间；缺省由服务端补 |

## 4. 响应

成功：
```json
{"accepted": true, "dedup": false, "dest": "data/feedback/2026-09-14.jsonl"}
```
重复推送（sn 已存在）：
```json
{"accepted": false, "dedup": true, "reason": "sn 已存在"}
```
参数错误（HTTP 400）：`{"detail": "human_class 需在 0-14"}`

## 5. 类目表（15 类，0-14）

| idx | 类目 | | idx | 类目 |
|---|---|---|---|---|
| 0 | 文件证件 | | 8 | 图书文娱 |
| 1 | 服饰鞋包 | | 9 | 运动户外 |
| 2 | 数码家电 | | 10 | 汽配五金工业品 |
| 3 | 食品生鲜 | | 11 | 家具家装 |
| 4 | 日用家居 | | 12 | 易碎品 |
| 5 | 美妆个护 | | 13 | 违禁限寄 |
| 6 | 医药保健 | | 14 | 其他/拒识 |
| 7 | 母婴玩具 | | | |

## 6. 调用示例

```bash
curl -X POST http://127.0.0.1:8004/feedback \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your_key>" \
  -d '{"sn":"h_001","text":"小猫","human_class":13,"model_pred":4,"model_conf":0.9393,
       "model_ver":"v2-distill-soft-e3","human_changed":true,"human_note":"活体动物",
       "channel":"web","ts":"2026-09-14T21:00:00+08:00"}'
```

Python：
```python
import json, urllib.request

body = json.dumps({
    "sn": "h_001", "text": "小猫", "human_class": 13, "model_pred": 4,
    "model_conf": 0.9393, "model_ver": "v2-distill-soft-e3",
    "human_changed": True, "channel": "web",
}).encode("utf-8")
req = urllib.request.Request("http://<host>:8004/feedback", data=body,
                             headers={"Content-Type": "application/json",
                                      "X-API-Key": "<your_key>"})
print(json.loads(urllib.request.urlopen(req).read()))
```

## 7. 部署对接前提（我方侧）

- 当前服务监听 `127.0.0.1:8004`（本机调试态）。**真实对接需把服务部署到服务器并对外暴露端口**（或公司网络可达的地址），并设置 `API_AUTH_KEY` 环境变量启用鉴权。
- 数据按天落盘为 `data/feedback/YYYY-MM-DD.jsonl`，我方每周清洗聚合用于模型迭代。

## 8. FAQ

- **改判了但没填原因行不行？** 行。原因选填，但填了价值更高（直接指导词库补丁）。
- **同一个单被复检两次怎么算？** 按 `sn` 去重，只保留第一条；后续清洗按同文本多数票聚合。
- **多副本扩容时一致性注意**：当前幂等（sn 集合）与限流均为**进程内**实现——`docker compose --scale` 扩到多副本后，各副本独立去重与计数（同一 sn 可能在多副本各收一次、限流按每副本计）。真实对接若需扩容：建议限流前置到网关/Nginx 层，或接入共享存储（Redis 等）做全局 sn 去重。
- **`text` 能不能发清洗后的？** 不能，要原文——方言、错别字、口语正是模型要学的分布。
- **数据会对外公开吗？** 不会，仅内部用于模型迭代；字段已最小化且不含个人敏感信息。