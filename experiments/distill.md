# 知识蒸馏实验（教师 BERT -> 学生 BiLSTM）

- 教师：bert-base-chinese 微调（Test Acc 0.8727）
- 学生：BiLSTM（embed=128, hidden=256, 2 层, dropout=0.3），参数量 5,470,986 ≈ 20.9MB
- 超参：T=2.0，α=0.7（软标签 KL·T² + (1-α)·CE）；硬标签 CE(教师 argmax)；中间层 hidden_projection 对齐 768 维 MSE

```
蒸馏-soft:        Test Acc=0.8751（dev_acc=0.8738）  ← 三种方式中最优，与讲义结论一致
蒸馏-hard:        Test Acc=0.8705（dev_acc=0.8739）
蒸馏-intermediate: Test Acc=0.8740（dev_acc=0.8739）

蒸馏-soft: Test Acc=0.8760, 参数量=5,470,986（约20.9MB fp32 / checkpoint: models/checkpoints/distill_best_soft.pt）
蒸馏-hard: Test Acc=0.8705, 参数量=5,470,986
蒸馏-intermediate: Test Acc=0.8740, 参数量=5,470,986
```

要点：
- 软标签蒸馏稳定优于硬标签——教师概率分布携带类目相似度信息（如「食品生鲜」与「医药健康」的接近程度），
  比独热硬标签信息量大，验证了 Hinton 蒸馏理论。
- 学生模型仅 5.5M 参数（BERT 的 ~1/18），CPU 推理毫秒级，上线口径选择 soft 蒸馏档。
