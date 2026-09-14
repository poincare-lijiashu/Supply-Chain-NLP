# 知识蒸馏实验（教师 BERT -> 学生 BiLSTM · v2.300k-d2 · 2026-09-14）

```
蒸馏-soft(BERT_INIT=1): Test Acc=0.9980, 参数量=5,473,551（约20.9MB fp32）
checkpoint: models/checkpoints_v2/distill_best_soft.pt（备份 distill_best_soft_ep3_v09972.pt）
六尺验收: dev 0.9981 / test-ID 0.9980 / SIM 0.9944 / 违禁召回 0.9928 —— 详见 README §4 与 scripts/eval_student_v2.py
```
