# 模型压缩实验（量化 / 剪枝 / 延迟基准）

```
动态量化(int8, Linear) Test Acc=0.9637, Macro F1=0.9660
模型体积: fp32 390.2MB -> int8 145.6MB
量化模型: D:\workspace_AI\workspace_traecode\xiangmu_traecode\Supply Chain NLP\models\checkpoints\bert_best_int8.pt
              precision    recall  f1-score   support

        文件资料     1.0000    0.9600    0.9796      1000
        服饰鞋帽     1.0000    0.9580    0.9785      1000
        数码家电     1.0000    0.9590    0.9791      1000
        食品生鲜     0.9990    0.9600    0.9791      1000
        日用百货     1.0000    0.9600    0.9796      1000
        美妆个护     0.7364    1.0000    0.8482      1000
        医药健康     1.0000    0.9600    0.9796      1000
         易碎品     1.0000    0.9600    0.9796      1000
         违禁品     1.0000    0.9600    0.9796      1000
          其他     0.9959    0.9600    0.9776      1000

    accuracy                         0.9637     10000
   macro avg     0.9731    0.9637    0.9660     10000
weighted avg     0.9731    0.9637    0.9660     10000


L1剪枝30%（encoder.query）: Acc 0.9640->0.9640, F1 0.9664->0.9664, 稀疏度=0.3000
剪枝模型: D:\workspace_AI\workspace_traecode\xiangmu_traecode\Supply Chain NLP\models\checkpoints\bert_best_pruned.pt

BERT fp32 (CPU): 15.9 ms/条
BERT int8 (CPU): 10.3 ms/条
加速比: 1.53x
```
