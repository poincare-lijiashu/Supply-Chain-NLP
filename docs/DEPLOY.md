# 部署指南（DEPLOY）

> 适用版本：v1.2.0。两条路径：**Docker Compose（推荐）** 与 **裸机 uvicorn**。
> 全部命令在项目根目录执行。

## 0. 前置条件

| 项 | 要求 |
|---|---|
| Python | 3.11+（裸机方式） |
| Docker | 20.10+ 与 docker compose v2（容器方式） |
| 模型资产 | 镜像/服务只依赖蒸馏学生权重（21MB）+ BERT 分词器（~400KB），已在仓库内 |
| 端口 | 默认 8004（API）、18000（LB profile）；可自行映射 |

## 1. 方式一：Docker Compose（推荐）

```bash
# 构建并启动（单实例，容器网络内 8004，无固定宿主端口）
docker compose up -d --build
docker compose port api 8004        # 查看随机映射端口

# 健康检查
curl http://localhost:<端口>/health
```

**生产入口（Nginx 负载均衡 + 安全响应头）**：

```bash
docker compose --profile lb up -d          # 默认 1 副本
docker compose --profile lb up -d --scale api=4   # 水平扩容到 4 副本（轮询 LB）
curl http://localhost:18000/health
```

**鉴权（强烈建议在暴露到非本机网络时开启）**：

```bash
# 方式 A：环境变量
API_AUTH_KEY=<你的密钥> docker compose --profile lb up -d
# 方式 B：写入 .env（compose 自动读取，见 .env.example）
```

开启后所有 `/predict` 调用必须带请求头：

```bash
curl -X POST http://localhost:18000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <你的密钥>" \
  -d '{"texts": ["寄两份合同文件", "一箱车厘子"]}'
```

**切换模型**：`MODEL_NAME=distill|bert_int8|bert`（compose `environment` 或环境变量）。
`bert_int8` 需先在本机跑过 `python -m src.compress.quantize_prune` 生成 `_int8.pt` 并打进镜像。

**升级与回滚**：

```bash
docker compose down
docker build -t supply-chain-nlp:1.2.0 .        # 新版本构建
# 修改 docker-compose.yml 中 image tag 后
docker compose --profile lb up -d
# 回滚 = 把 image tag 改回旧版本再 up（镜像不可变，天然支持回滚）
```

## 2. 方式二：裸机 uvicorn

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows（Linux: source .venv/bin/activate）
pip install -r requirements.txt   # torch 按 https://pytorch.org 选择对应 CUDA/CPU wheel

uvicorn src.serving.app:app --host 127.0.0.1 --port 8004      # 默认蒸馏模型，仅本机
API_AUTH_KEY=<密钥> uvicorn src.serving.app:app --port 8004   # 开启鉴权
```

> 安全默认值：`serve_host=127.0.0.1`，只监听本机回环。需要局域网/公网访问时，
> **不要**直接改 0.0.0.0 裸奔——用容器 + Nginx（或网关）暴露，并开启 `API_AUTH_KEY`。

**Linux 常驻（systemd 示例）** `/etc/systemd/system/scnlp.service`：

```ini
[Unit]
Description=Supply Chain NLP API
After=network.target

[Service]
User=www-data
WorkingDirectory=/opt/supply-chain-nlp
Environment=MODEL_NAME=distill
Environment=API_AUTH_KEY=change-me
ExecStart=/opt/supply-chain-nlp/.venv/bin/uvicorn src.serving.app:app --host 127.0.0.1 --port 8004
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now scnlp
```

## 3. 接口与行为

| 端点 | 方法 | 鉴权 | 说明 |
|---|---|---|---|
| `/health` | GET | 否 | 存活探针：返回模型名与类目表 |
| `/predict` | POST | 配置后必须 | 单条（字符串）或批量（列表 ≤100 条，单条 ≤512 字符） |
| `/` 与 `/static/*` | GET | 否 | 前端工作台（纯静态，无 CDN 依赖） |

**响应约定**：每条文本返回 `class_name / prob / probs(10类) / needs_human_review / review_reason`；
`prob < 0.60` 或命中「违禁品」→ `needs_human_review=true`（红线策略）。

**错误码**：401（缺/错 X-API-Key）、429（超限流 60 次/分钟/客户端）、400（入参非法或超上限）。

## 4. 安全设计（已内置）

- **鉴权**：`API_AUTH_KEY` 环境变量 + `X-API-Key` 恒时比较（防时序侧信道）；留空 = 本机开放模式
- **限流**：内存滑动窗口按客户端 IP 统计（Nginx 场景已透传 `X-Real-IP`）；多副本下为每进程独立窗口，如需全局精确限流可接 Redis（当前单进程模型单例架构下无必要）
- **入参上限**：条数/长度硬校验，防资源滥用
- **权重安全**：全部 `torch.load(..., weights_only=True)`；量化产物保存 `state_dict`（非整模型 pickle）
- **容器**：非 root（uid 10001）、无状态、资源限额（compose 中 2 CPU / 2G）
- **Nginx**：`server_tokens off` + CSP/X-Frame-Options/X-Content-Type-Options/Referrer-Policy

**生产额外建议（按需）**：TLS 终止（Nginx/网关层挂证书）、密钥托管（Vault/云 KMS 而非 .env）、
公网入口加 WAF/认证网关、日志脱敏与留存策略。

## 5. 验证清单

```bash
pytest -q                       # 单测：核心算法 + 数据/配置
python -m scripts.run_smoke     # 全链路冒烟（产物隔离 data/smoke、models/smoke）
python -m src.serving.client    # 对着已启动的服务跑一遍真实预测
```

## 6. 常见问题

- **端口被占**：改 compose `ports` 映射或 `--port`；确认 `netstat -ano | findstr 8004`（Windows）
- **强制 CPU 推理**：设置环境变量 `MODEL_DEVICE=cpu`（Windows 系统内存吃紧时 CUDA 可能报"假 OOM"——显存充足仍报错，强制 CPU 最稳；容器内 CPU 推理不受影响）
- **模型文件缺失报错**：`MODEL_NAME=bert_int8/bert` 需要对应 checkpoint 先训练产出；默认 `distill` 只需仓库内 `distill_best_soft.pt`
- **Windows 防火墙**：本机回环访问不需要放行；仅容器端口映射访问异常时检查 Docker Desktop 网络配置
- **首次启动慢**：加载模型单例约数秒，`HEALTHCHECK` 已设 `start-period=20s`
