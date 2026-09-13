# 模型压缩实验（量化 / 剪枝 / 延迟基准）

```
动态量化(int8, Linear) Test Acc=0.9652, Macro F1=0.9672
模型体积: fp32 390.2MB -> int8 145.6MB
量化模型: D:\workspace_AI\workspace_traecode\xiangmu_traecode\Supply Chain NLP\models\checkpoints\bert_best_int8.pt
              precision    recall  f1-score   support

        文件资料     1.0000    0.9591    0.9791       979
        服饰鞋帽     1.0000    0.9593    0.9792       982
        数码家电     0.7817    0.9942    0.8752      1037
        食品生鲜     0.9955    0.9602    0.9776       930
        日用百货     1.0000    0.9624    0.9808      1063
        美妆个护     1.0000    0.9577    0.9784       921
        医药健康     1.0000    0.9622    0.9807       979
         易碎品     0.9618    0.9717    0.9668      1167
         违禁品     1.0000    0.9604    0.9798      1010
          其他     0.9879    0.9614    0.9744       932

    accuracy                         0.9652     10000
   macro avg     0.9727    0.9649    0.9672     10000
weighted avg     0.9714    0.9652    0.9667     10000


L1剪枝30%（encoder.query）: Acc 0.9650->0.9649, F1 0.9673->0.9672, 稀疏度=0.3000
剪枝模型: D:\workspace_AI\workspace_traecode\xiangmu_traecode\Supply Chain NLP\models\checkpoints\bert_best_pruned.pt

BERT fp32 (CPU): 19.5 ms/条
BERT int8 (CPU): 13.2 ms/条
加速比: 1.47x
```
