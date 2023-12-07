总结：
做了意图检测还有SLOT填充，做了储存以单词 word 、标注 slot 、目的 intent的字典可查询修改，做了训练数据、测试数据。模型做了编码器（LSTM）和解码器。SLOT filling的准确率和F1详细见附录运行结果图。
问题：
在PC上的环境是Win10 + Pytorch 0.1 + Python 3.6，可能因为部分包的原因有一些warnings。太大的 batch 会拖慢训练速度，很关键。
23秋季班大数据与科学  李琛
