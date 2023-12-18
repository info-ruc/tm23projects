基于BERT模型进行中文地址的分类识别
训练数据使用<neural-chinese-address-parsing>

bert_local_trans.ipynb、project.ipynb项目主体文件

bert_local_trans.ipynb
这段代码调用 Hugging Face 的 transformers 库和 datasets 库实现了一个中文地址分类识别的语言模型训练。并输出<Neural_Chinese_Address_Parsing_BERT_state_dict.pkl>文件来保存训练模型。

project.ipynb
这段代码主要功能是对键盘输入的一段中文地址进行分类识别并输出。这段代码与'bert_local_trans.ipynb'调用了相同的库，并载入了<Neural_Chinese_Address_Parsing_BERT_state_dict.pkl>模型参数。同时为了兼容性使用了cpu进行运算。


