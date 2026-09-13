# 模型压缩实验（量化 / 剪枝 / 延迟基准）

```
动态量化(int8, Linear) Test Acc=0.8728, Macro F1=0.8827
模型体积: fp32 390.2MB -> int8 145.6MB
量化模型: D:\workspace_AI\workspace_traecode\xiangmu_traecode\Supply Chain NLP\models\checkpoints\bert_best_int8.pt
              precision    recall  f1-score   support

        文件资料     1.0000    0.8590    0.9242      1000
        服饰鞋帽     1.0000    0.8600    0.9247      1000
        数码家电     0.6596    0.9050    0.7631      1000
        食品生鲜     0.5668    0.9290    0.7041      1000
        日用百货     0.9988    0.8600    0.9242      1000
        美妆个护     1.0000    0.8600    0.9247      1000
        医药健康     0.9740    0.8630    0.9152      1000
         易碎品     1.0000    0.8600    0.9247      1000
         违禁品     1.0000    0.8600    0.9247      1000
          其他     0.9247    0.8720    0.8976      1000

    accuracy                         0.8728     10000
   macro avg     0.9124    0.8728    0.8827     10000
weighted avg     0.9124    0.8728    0.8827     10000


L1剪枝30%（encoder.query）: Acc 0.8735->0.8729, F1 0.8875->0.8860, 稀疏度=0.3000
剪枝模型: D:\workspace_AI\workspace_traecode\xiangmu_traecode\Supply Chain NLP\models\checkpoints\bert_best_pruned.pt

BERT fp32 (CPU): 25.6 ms/条
BERT int8 (CPU): 14.6 ms/条
加速比: 1.75x
```
