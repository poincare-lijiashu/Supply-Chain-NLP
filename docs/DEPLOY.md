# 部署指南（DEPLOY）

> 适用版本：v2（15 类全链路，蒸馏学生上线）+ 兼容 v1 路径说明。两条路径：**Docker Compose（推荐）** 与 **裸机 uvicorn**。
> 全部命令在项目根目录执行；数据/模型统一 `DATA_VERSION=v2`（v1 已归档 `data/raw_v1/`，如需 v1 服务设 `DATA_VERSION=v1`）。

## 0. 前置条件

| 项 | 要求 |
|---|---|
| Python | 3.11+（裸机方式） |
| Docker | 20.10+ 与 docker compose v2（容器方式） |
| 模型资产 | 蒸馏学生权重 `models/checkpoints_v2/distill_best_soft.pt`（21MB）+ BERT 分词器（~400KB）+ 护栏词表（`oov_vocab.json` / `redline_words.json`），已在仓库生成脚本可再产 |
| 类目表 | `data/raw_v2/class.txt`（15 类） |
| 端口 | 默认 8004（API）、18000（LB profile）；可自行映射 |

## 1. 方式一：Docker Compose（推荐）

```bash
# 构建并启动（DATA_VERSION=v2 已内置 Dockerfile；容器网络内 8004，无固定宿主端口）
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

**持久化**：`./data/feedback` 挂载为卷（compose）——快递员复检终判回流在容器重建后不丢。

**鉴权（强烈建议在暴露到非本机网络时开启）**：

```bash
# 方式 A：环境变量
API_AUTH_KEY=<你的密钥> FEEDBACK_AUTH_KEY=<回流专用密钥> docker compose --profile lb up -d
# 方式 B：写入 .env（compose 自动读取，见 .env.example）
```

开启后所有 `/predict` 调用必须带请求头：

```bash
curl -X POST http://localhost:18000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <你的密钥>" \
  -d '{"texts": ["寄两份合同文件", "一箱车厘子"]}'
```

**回流接口** `/feedback` 同端口：`X-API-Key` 用 `FEEDBACK_AUTH_KEY`（未设置则回退 `API_AUTH_KEY`），
限流独立（120 次/分/客户端）。契约见 [DATA_FEEDBACK.md](DATA_FEEDBACK.md)。

**切换模型**：`MODEL_NAME=distill|bert|bert_int8`（compose `environment` 或环境变量）。
默认 `distill`（v2 学生）；`bert` 加载 `checkpoints_v2/bert_best.pt`（390MB，需打进镜像）。

**升级与回滚**：

```bash
docker compose down
docker build -t supply-chain-nlp:1.3.0 .        # 新版本构建
# 修改 docker-compose.yml 中 image tag 后
docker compose --profile lb up -d
# 回滚 = 把 image tag 改回旧版本再 up（镜像不可变，天然支持回滚）
```

## 2. 方式二：裸机 uvicorn

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows（Linux: source .venv/bin/activate）
pip install -r requirements.txt   # torch 按 https://pytorch.org 选择对应 CUDA/CPU wheel

DATA_VERSION=v2 uvicorn src.serving.app:app --host 127.0.0.1 --port 8004      # 默认蒸馏学生，仅本机
API_AUTH_KEY=<密钥> FEEDBACK_AUTH_KEY=<回流密钥> DATA_VERSION=v2 uvicorn src.serving.app:app --port 8004
```

> 安全默认值：`serve_host=127.0.0.1`，只监听本机回环。需要局域网/公网访问时，
> **不要**直接改 0.0.0.0 裸奔——用容器 + Nginx（或网关）暴露，并开启鉴权。

**Linux 常驻（systemd 示例）** `/etc/systemd/system/scnlp.service`：

```ini
[Unit]
Description=Supply Chain NLP API
After=network.target

[Service]
User=www-data
WorkingDirectory=/opt/supply-chain-nlp
Environment=DATA_VERSION=v2
Environment=MODEL_NAME=distill
Environment=API_AUTH_KEY=change-me
Environment=FEEDBACK_AUTH_KEY=change-me-feedback
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
| `/health` | GET | 否 | 存活探针：返回模型名、类目表、OOV 护栏开关 |
| `/predict` | POST | 配置后必须 | 单条（字符串）或批量（列表 ≤100 条，单条 ≤512 字符） |
| `/feedback` | POST | `FEEDBACK_AUTH_KEY` 或 `API_AUTH_KEY` | 快递员复检终判回流（sn 幂等 + 限流 + 上限，见 DATA_FEEDBACK.md） |
| `/` 与 `/static/*` | GET | 否 | 前端工作台（纯静态，无 CDN 依赖） |

**响应约定**：每条文本返回 `class_name / prob / probs(15类) / needs_human_review / review_reason`。
转人工判定（多道独立护栏，任一命中即 `needs_human_review=true`）：
- `prob < 0.60`（低置信度）
- 命中「违禁限寄」类目（红线类目）
- 命中禁寄词库（`redline_words.json`，与置信度无关——堵"模型高置信错分"）
- 未登录词 + 中间带置信（0.60~0.95，OOV 护栏）

**错误码**：401（缺/错 X-API-Key）、429（超限流：predict 60 次/分、feedback 120 次/分，或反馈达 `FEEDBACK_MAX_SN` 上限）、400（入参非法或超上限）。

## 4. 安全设计（已内置）

- **鉴权**：`/predict` 用 `API_AUTH_KEY`；`/feedback` 独立 `FEEDBACK_AUTH_KEY`（可分开管理）；恒时比较防时序侧信道；留空 = 本机开放模式
- **限流**：内存滑动窗口按客户端 IP 统计（Nginx 场景已透传 `X-Real-IP`）；predict 与 feedback 独立桶；feedback 幂等集合上限 `FEEDBACK_MAX_SN`（默认 10 万）防无界增长
- **入参上限**：条数/长度硬校验，防资源滥用
- **去反序列化面**：全部 `torch.load(..., weights_only=True)`；护栏词表为 JSON 纯数据（无 pickle）
- **容器**：非 root（uid 10001）、无状态、资源限额（compose 中 2 CPU / 2G）
- **Nginx**：`server_tokens off` + CSP/X-Frame-Options/X-Content-Type-Options/Referrer-Policy

**生产额外建议（按需）**：TLS 终止（Nginx/网关层挂证书）、密钥托管（Vault/云 KMS 而非 .env）、
公网入口加 WAF/认证网关、日志脱敏与留存策略（/feedback 只收最小字段、无 PII）。

## 5. 验证清单

```bash
pytest -q                       # 单测：62 passed（核心算法 + v2 数据/配置 + 安全 + 生成器规则）
python -m scripts.eval_student_v2   # 学生六尺上报（上线前回归）
python -m src.serving.client    # 对着已启动的服务跑一遍真实预测
```

## 6. 常见问题

- **端口被占**：改 compose `ports` 映射或 `--port`；确认 `netstat -ano | findstr 8004`（Windows）
- **强制 CPU 推理**：设置环境变量 `MODEL_DEVICE=cpu`（Windows 系统内存吃紧时 CUDA 可能报"假 OOM"——显存充足仍报错，强制 CPU 最稳；容器内 CPU 推理不受影响）
- **护栏词表缺失**：OOV 词表缺失 → **fail-closed**（未登录词无法判定，中间带样本全部转人工，见 serving 日志告警后立即重建词表）；红线词库缺失 → fail-open（仅损失禁寄词强拦）。两个词表均可由 `scripts/build_oov_vocab.py` / `scripts/build_redline_words.py` 再生成
- **模型文件缺失报错**：`MODEL_NAME=bert` 需要 `checkpoints_v2/bert_best.pt` 先训练产出；默认 `distill` 只需 `checkpoints_v2/distill_best_soft.pt`
- **Windows 防火墙**：本机回环访问不需要放行；仅容器端口映射访问异常时检查 Docker Desktop 网络配置
- **首次启动慢**：加载模型单例约数秒，`HEALTHCHECK` 已设 `start-period=20s`