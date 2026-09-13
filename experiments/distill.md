# 知识蒸馏实验（教师 BERT → 学生 BiLSTM）

- 教师：bert-base-chinese 微调（难度基准 v1.3.1 · 4 epochs，Test Acc 96.50%，Macro F1 0.9673）
- 学生：BiLSTM（embed=128, hidden=256, 双向 2 层, dropout=0.3），参数量 5,470,986 ≈ 20.9MB（教师约 1/19）
- 超参：T=2.0，α=0.7（软标签 KL·T² + (1-α)·CE）；6 epochs（难度提升后学生收敛需更多轮次）

```
蒸馏-soft（难度基准 v1.3.1 · 6 epochs）:
  Test Acc=0.9650, Macro F1=0.9668 —— 与教师（96.50%）打平（差异在 ±0.1pp 评估波动内）
  checkpoint: models/checkpoints/distill_best_soft.pt
```

要点：
- **学生追平教师**：软标签携带类目相似度信息（如「食品生鲜」与「医药健康」的接近程度），
  等效于标签平滑正则，学生在 1/19 体积下泛化与教师打平——Hinton 蒸馏理论的经典现象
  （蒸馏学生超越/追平教师在文献中可复现，如 Born-Again Networks；本例差异在波动内，
  宜解读为"打平"）
- **压缩收益**：体积 1/19（20.9MB vs 397MB），CPU 推理 1.3ms（约 15×），精度无损
- 复现：`python -m src.compress.distill`（soft/hard/intermediate 三种方式全跑）；
  soft 单跑见 `src.compress.distill.distill(conf, mode="soft")`
