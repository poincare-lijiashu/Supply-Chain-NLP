# 物流寄件文本智能分类服务 —— CPU 推理镜像（脱离本机环境，开箱即用）
# 构建：docker build -t supply-chain-nlp:1.2.0 .
# 运行：docker run -d -p 18004:8004 --name supply-chain-nlp supply-chain-nlp:1.2.0
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_NAME=distill \
    DATA_VERSION=v2

WORKDIR /app

# 依赖层（CPU 版 torch；与 requirements.txt 同版本）
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple \
 && pip install --no-cache-dir "uvicorn[standard]" \
 # fasttext 0.9.x 与 numpy 2.x 兼容补丁（与 docs/fix_log.md 一致）
 && python -c "import fasttext, pathlib; p = pathlib.Path(fasttext.__file__).parent / 'FastText.py'; s = p.read_text(encoding='utf-8'); p.write_text(s.replace('np.array(probs, copy=False)', 'np.asarray(probs)'), encoding='utf-8'); print('fasttext numpy2 patch ok')"

# 代码与配置
COPY config/ config/
COPY src/ src/
COPY scripts/ scripts/

# 运行时资产：分词器（~400KB）+ 蒸馏学生权重（21MB）+ 类目/停用词 + 护栏词表
COPY models/bert-base-chinese/config.json models/bert-base-chinese/vocab.txt models/bert-base-chinese/tokenizer_config.json models/bert-base-chinese/tokenizer.json models/bert-base-chinese/
COPY models/checkpoints_v2/distill_best_soft.pt models/checkpoints_v2/
COPY data/raw_v2/class.txt data/raw_v2/
COPY data/stopwords.txt data/
COPY data/processed/oov_vocab.json data/processed/redline_words.json data/processed/
COPY web/ web/

# 非 root 运行（镜像内只读服务，无写需求）
RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8004
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8004/health')" || exit 1

# uvicorn 单进程加载模型单例；多实例扩容交给 docker compose --scale 或 K8s
CMD ["python", "-m", "uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8004"]
