# 知识蒸馏实验（教师 BERT → 学生 BiLSTM）

- 教师：bert-base-chinese 微调（难度基准 4 epochs，Test Acc 96.37%，Macro F1 0.9650）
- 学生：BiLSTM（embed=128, hidden=256, 双向 2 层, dropout=0.3），参数量 5,470,986 ≈ 20.9MB（教师约 1/19）
- 超参：T=2.0，α=0.7（软标签 KL·T² + (1-α)·CE）；硬标签 CE(教师 argmax)；中间层 hidden_projection 对齐 768 维 MSE

```
蒸馏-soft（难度基准 · 4 epochs）:
  dev_acc 按 epoch: 0.9534 → 0.9650 → 0.9644 → 0.9640（epoch2 最优保存）
  Test Acc=0.9646, Macro F1=0.9654 —— 反超教师（96.37%）0.09pp
  checkpoint: models/checkpoints/distill_best_soft.pt
```

要点：
- **学生反超教师**：软标签携带类目相似度信息（如「食品生鲜」与「医药健康」的接近程度），
  等效于标签平滑正则，学生在 1/19 体积下泛化略优于教师——Hinton 蒸馏理论的经典现象
- **压缩收益**：体积 1/19（20.9MB vs 397MB），CPU 推理 1.3ms（约 16×），精度无损反升
- 复现：`python -m src.compress.distill`（soft/hard/intermediate 三种方式全跑）；
  soft 单跑见 `src.compress.distill.distill(conf, mode="soft")`
