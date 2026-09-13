# 模型压缩实验（量化 / 剪枝 / 延迟基准）

```
动态量化(int8, Linear) Test Acc=0.9648, Macro F1=0.9673
模型体积: fp32 390.2MB -> int8 145.6MB
量化模型: D:\workspace_AI\workspace_traecode\xiangmu_traecode\Supply Chain NLP\models\checkpoints\bert_best_int8.pt
              precision    recall  f1-score   support

        文件资料     1.0000    0.9591    0.9791       979
        服饰鞋帽     1.0000    0.9593    0.9792       982
        数码家电     1.0000    0.9595    0.9793      1037
        食品生鲜     1.0000    0.9591    0.9791       930
        日用百货     0.7641    0.9962    0.8648      1063
        美妆个护     0.9977    0.9577    0.9773       921
        医药健康     1.0000    0.9622    0.9807       979
         易碎品     0.9991    0.9657    0.9821      1167
         违禁品     0.9789    0.9644    0.9716      1010
          其他     0.9989    0.9614    0.9798       932

    accuracy                         0.9648     10000
   macro avg     0.9739    0.9645    0.9673     10000
weighted avg     0.9724    0.9648    0.9666     10000


L1剪枝30%（encoder.query）: Acc 0.9652->0.9652, F1 0.9676->0.9676, 稀疏度=0.3000
剪枝模型: D:\workspace_AI\workspace_traecode\xiangmu_traecode\Supply Chain NLP\models\checkpoints\bert_best_pruned.pt

BERT fp32 (CPU): 16.9 ms/条
BERT int8 (CPU): 12.7 ms/条
加速比: 1.33x
```
