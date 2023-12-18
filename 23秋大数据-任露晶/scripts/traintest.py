#第6章/加载tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('hfl/rbt3')
out = tokenizer.batch_encode_plus(
['你站在桥上看风景',
'看风景的人在楼上看你',
'明月装饰了你的窗子',
'你装饰了别人的梦'],
truncation=True,
)
print(out)

#第6章/从磁盘加载数据集
from datasets import load_dataset
dataset = load_dataset("seamew/ChnSentiCorp")
#缩小数据规模，便于测试
dataset['train'] = dataset['train'].shuffle().select(range(2000))
dataset['test'] = dataset['test'].shuffle().select(range(100))
print(dataset)