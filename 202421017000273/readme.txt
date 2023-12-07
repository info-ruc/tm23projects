总结：
做了意图检测还有SLOT填充，储存以单词 word 、标注 slot 、目的 intent的字典，可查询修改，模型做了编码器（LSTM）和解码器。SLOT filling任务的准确率0.896996和F1得分0.713193；intentiondetection的任务准确率0.865731
问题：
在PC上的环境是Win10 + Pytorch 0.1 + Python 3.6，可能因为部分包的原因有一些warnings。太大的 batch 会拖慢训练速度，很关键。
