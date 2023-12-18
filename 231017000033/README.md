# Build development environment

## 1. Prepare development environment
- checkout system version
```
uname -a
```

- checkout python version
```
python3 --version
```
- my system version
```
Linux reikocao 6.2.0-37-generic #38~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Thu Nov  2 18:01:13 UTC 2 x86_64 x86_64 x86_64 GNU/Linux
```
- my python version
```
Python 3.10.12
```

## 2. Install dependencies
- neccessary
```
pip install 'portalocker>=2.0.0'
pip3 install torchtext
pip3 install torch
pip3 install torchdata
pip3 install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 3. Install torch via whl file when the command "pip3 install torch" is failed
- note that the library version must be suitable for your development environment, search them in the websit pypi.
```
https://pypi.org/
```

- my whl file links
```
https://files.pythonhosted.org/packages/96/82/0966469ded5946cb4c18dd11b04eac78c943269fc79d290740d6477005e8/torch-2.1.1-cp310-cp310-manylinux1_x86_64.whl
https://files.pythonhosted.org/packages/4d/22/91a8af421c8a8902dde76e6ef3db01b258af16c53d81e8c0d0dc13900a9e/triton-2.1.0-0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
```
- install whl file
```
pip3 install torch-2.1.1-cp310-cp310-manylinux1_x86_64.whl
pip3 install triton-2.1.0-0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
```

# Run project

## 1. Run script
```
python3 classification.py
```

## 2. Model print
```
NormalNet(
  (embedding): EmbeddingBag(95811, 64, mode='mean')
  (fc): Linear(in_features=64, out_features=4, bias=True)
)
```

## 2. The screenshot of result 
```
the screenshots are saved in the folder 'screenshot'
```

# Additional
## Train transformer model
```
python3 ./utils/transformer.py
```

# Basic conception

## LLM
- [大型语言模型（Large Language Models，LLMs）概览](https://zhuanlan.zhihu.com/p/639318309)
- [Transformer学习笔记一：Positional Encoding（位置编码）](https://zhuanlan.zhihu.com/p/454482273)
- [时序模型系列——最详细的Transformer模型解析](https://zhuanlan.zhihu.com/p/569277709)
- [Transformer模型（总结）](https://zhuanlan.zhihu.com/p/473236819)
- [Transformers代码解读——Bert输入向量](https://zhuanlan.zhihu.com/p/631315484)

## 1. Embedding
- [深入理解PyTorch中的nn.Embedding](https://blog.csdn.net/raelum/article/details/125462028)

## 2. Dataset
- [pytorch dataset](https://github.com/pytorch/data)
- [pytorch dataset specification](https://pytorch.org/text/stable/datasets.html)
- [PyTorch 自定义数据集](https://zhuanlan.zhihu.com/p/608090658?utm_id=0)
- [pytorch 构建自己的文本分类dataset](https://zhuanlan.zhihu.com/p/440558197)

## 3. Example
- [NLP实战：中文文本分类-Pytorch实现](https://blog.csdn.net/m0_62237233/article/details/130887570)