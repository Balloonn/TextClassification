# TextClassification

### 词向量来源
中文词向量参考网站： https://github.com/Embedding/Chinese-Word-Vectors

使用Wikipedia_zh中文维基百科的Word + Character词向量文件sgns.wiki.char.bz2，词向量的维度为300。
https://pan.baidu.com/s/1ZBVVD4mUSUuXOxlZ3V71ZA

### 文件运行顺序
1. preprocessing.py 数据预处理，生成词典与词向量pk文件
2. get_pretrained_vector.py 加载经过预训练的词向量
3. model_train.py 训练模型
4. model_eval.py 绘图查看训练准确率等信息