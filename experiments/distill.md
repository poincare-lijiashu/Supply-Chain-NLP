# 知识蒸馏实验（教师 BERT → 学生 BiLSTM）

- 教师：bert-base-chinese 微调（治理后数据 4 epochs，Test Acc 0.9638，Macro F1 0.9660）
- 学生：BiLSTM（embed=128, hidden=256, 双向 2 层, dropout=0.3），参数量 5,470,986 ≈ 20.9MB（教师约 1/19）
- 超参：T=2.0，α=0.7（软标签 KL·T² + (1-α)·CE）；硬标签 CE(教师 argmax)；中间层 hidden_projection 对齐 768 维 MSE

```
蒸馏-soft（最新一轮 · 治理后数据 · 4 epochs）:
  dev_acc 按 epoch: 0.9626 → 0.9571 → 0.9640 → 0.9637（最优保存）
  Test Acc=0.9637, 参数量=5,470,986（约20.9MB fp32 / checkpoint: models/checkpoints/distill_best_soft.pt）

历史对照（治理前数据 · 2 epochs，三种方式横向比较）:
  蒸馏-soft:         Test Acc=0.8760  ← 三种方式中最优
  蒸馏-hard:         Test Acc=0.8705
  蒸馏-intermediate: Test Acc=0.8740
```

要点：
- **soft 蒸馏几乎无损**：治理后数据上学生（96.37%）与教师（96.38%）仅差 0.01 个百分点
- 软标签优于硬标签/中间层（横向对照见上）——教师概率分布携带类目相似度信息（如「食品生鲜」与
  「医药健康」的接近程度），比独热硬标签信息量大，与 Hinton 蒸馏理论一致
- 学生仅 5.5M 参数，CPU 推理毫秒级（1.3ms/条），上线口径选择 soft 蒸馏档
- 复现：`python -m src.compress.distill`（soft/hard/intermediate 三种方式全跑）；
  soft 单跑见 `src.compress.distill.distill(conf, mode="soft")`
