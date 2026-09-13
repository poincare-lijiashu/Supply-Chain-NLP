# 模型压缩实验（量化 / 剪枝 / 延迟基准）

```
动态量化(int8, Linear) Test Acc=0.9640, Macro F1=0.9651
模型体积: fp32 390.2MB -> int8 145.6MB
量化模型: D:\workspace_AI\workspace_traecode\xiangmu_traecode\Supply Chain NLP\models\checkpoints\bert_best_int8.pt
              precision    recall  f1-score   support

        文件资料     0.9752    0.9619    0.9685      1023
        服饰鞋帽     1.0000    0.9605    0.9798      1012
        数码家电     1.0000    0.9600    0.9796      1001
        食品生鲜     1.0000    0.9598    0.9795       969
        日用百货     0.9990    0.9602    0.9792      1004
        美妆个护     0.9977    0.9589    0.9779       924
        医药健康     1.0000    0.9588    0.9790       947
         易碎品     1.0000    0.9660    0.9827      1234
         违禁品     0.7357    0.9968    0.8465       927
          其他     1.0000    0.9583    0.9787       959

    accuracy                         0.9640     10000
   macro avg     0.9708    0.9641    0.9651     10000
weighted avg     0.9726    0.9640    0.9662     10000


L1剪枝30%（encoder.query）: Acc 0.9637->0.9638, F1 0.9650->0.9652, 稀疏度=0.3000
剪枝模型: D:\workspace_AI\workspace_traecode\xiangmu_traecode\Supply Chain NLP\models\checkpoints\bert_best_pruned.pt

BERT fp32 (CPU): 20.5 ms/条
BERT int8 (CPU): 12.9 ms/条
加速比: 1.58x
```
