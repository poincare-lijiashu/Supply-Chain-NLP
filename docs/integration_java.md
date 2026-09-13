# 企业后端集成指南（Java 调用本模型服务）

本服务是**无状态 REST API**，与语言无关。公司 Java 后端（Spring Boot 等）通过 HTTP 调用即可，
不需要引入任何 Python 依赖。

## 服务接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/predict` | 分类：`{"texts": "..."}` 或 `{"texts": ["...", "..."]}`（批量） |
| GET | `/health` | 健康检查（接入公司监控/K8s 探针） |
| GET | `/docs` | Swagger 交互文档（内网调试用） |

响应（数组，每条输入一个元素）：

```json
[{
  "text": "充电宝2个",
  "class_index": 8,
  "class_name": "违禁品",
  "prob": 1.0,
  "probs": {"文件资料": 0.0, "...": 0.0, "违禁品": 1.0},
  "needs_human_review": true,
  "review_reason": "违禁品类目强制人工复核（寄递安全红线）"
}]
```

## Spring Boot 调用示例（RestClient，Spring 6.1+）

```java
// 1. 配置（application.yml）
// nlp:
//   classify:
//     base-url: http://supply-chain-nlp:8004   # docker 网络内用服务名；跨机用 IP:18004
//     timeout-ms: 3000

@Component
public class NlpClassifyClient {

    private final RestClient restClient;

    public NlpClassifyClient(@Value("${nlp.classify.base-url}") String baseUrl) {
        this.restClient = RestClient.builder()
                .baseUrl(baseUrl)
                .requestFactory(new JdkClientHttpRequestFactory(
                        HttpClient.newBuilder().connectTimeout(Duration.ofMillis(1000)).build()))
                .build();
    }

    /** 单条分类 */
    public ClassifyResult classify(String text) {
        return classifyBatch(List.of(text)).get(0);
    }

    /** 批量分类（推荐：一次请求携带整页运单，减少 RT 开销） */
    public List<ClassifyResult> classifyBatch(List<String> texts) {
        Map<String, Object> body = Map.of("texts", texts);
        ClassifyResult[] arr = restClient.post()
                .uri("/predict")
                .contentType(MediaType.APPLICATION_JSON)
                .body(body)
                .retrieve()
                .body(ClassifyResult[].class);
        return Arrays.asList(arr);
    }
}

// 2. 响应结构（与 API 字段一一对应）
public record ClassifyResult(
        String text,
        int classIndex,
        String className,
        double prob,
        Map<String, Double> probs,
        boolean needsHumanReview,
        String reviewReason) {}
```

旧版 Spring 可用 RestTemplate/OpenFeign 等价实现；非 Spring 项目用 HttpClient/OkHttp 均可。

## 生产实践建议

1. **超时与降级**：单条推理 P99 < 10ms（蒸馏模型 CPU），客户端超时建议 1~3s；超时/不可用时降级走原人工初审队列，不阻塞下单主链路。
2. **重试与熔断**：GET /health 幂等可重试；POST /predict 幂等（同输入同输出）可安全重试 1 次；接入 Resilience4j 熔断，连续失败自动切人工。
3. **批量优先**：一次运单页（20~50 条）合一个请求，吞吐最高。
4. **结果消费约定**：`needs_human_review=true` 的记录进人工复核队列（含 review_reason）；违禁品红线类目**不要**只信模型，必须复核。
5. **鉴权**：当前服务未内置鉴权，务必通过公司网关/内网隔离暴露，或在 nginx 层加 API Key / mTLS。
6. **链路追踪**：可在请求头带 `X-Request-Id`，排查问题时结合 uvicorn 访问日志。

## 消息队列形态（可选，削峰）

寄件高峰可在 Java 侧发 Kafka/RocketMQ 消息，由消费者批量调 `/predict` 后写回结果表——
服务本身无状态，消费组可水平扩容。
