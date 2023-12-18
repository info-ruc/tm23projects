from transformers import BertTokenizer
# 第一步 加载编码工具
# 参 数 pretrained_model_name_or_path='bert-basechinese'指定要加载的编码工具，大多数模型会把自己提交的编码工
# 具命名为和模型一样的名字。
# 模型和它的编码工具通常是成对使用的，不会出现张冠李戴的情
# 况，建议调用者也遵从习惯，成对使用。
# 参数cache_dir用于指定编码工具的缓存路径，这里指定为
# None（默认值），也可以指定想要的缓存路径。
# 参数force_download为True时表明无论是否已经有本地缓存，
# 都强制执行下载工作。建议设置为False。
tokenizer = BertTokenizer.from_pretrained(
pretrained_model_name_or_path='bert-base-chinese',
cache_dir=None,
force_download=False,
)
# 准备实验数据
sents = [
'你站在桥上看风景',
'看风景的人在楼上看你',
'明月装饰了你的窗子',
'你装饰了别人的梦',
'我站在桥上看风景',
'看风景的人在楼上看我',
'明月装饰了我的窗子',
'我装饰了别人的梦'
]
#基本的编码函数
# (1)参数text和text_pair分别为两个句子，如果只想编码一个句
# 子，则可让text_pair传None。
# (2)参数truncation=True表明当句子长度大于max_length时，
# 截断句子。
# (3) 参 数 padding= 'max_length' 表 明 当 句 子 长 度 不 足
# max_length时，在句子的后面补充PAD，直到max_length长度。
# (4)参数add_special_tokens=True表明需要在句子中添加特殊符
# 号。
# (5)参数max_length=25定义了max_length的长度。
# (6)参数return_tensors=None表明返回的数据类型为list格式，
# 也可 以 赋值为tf 、 pt 、 np ， 分 别 表 示 TensorFlow 、 PyTorch 、
# NumPy数据格式。
out = tokenizer.encode(
text=sents[0],
text_pair=sents[1],
#当句子长度大于max_length时截断
truncation=True,
#一律补PAD，直到max_length长度
padding='max_length',
add_special_tokens=True,
max_length=25,
return_tensors=None,
)
print(out)
print(tokenizer.decode(out))

###########################################################
#进阶的编码函数
out = tokenizer.encode_plus(
text=sents[0],
text_pair=sents[1],
#当句子长度大于max_length时截断
truncation=True,
#一律补零，直到max_length长度
padding='max_length',
max_length=25,
add_special_tokens=True,
#可取值tf、pt、np，默认为返回list
return_tensors=None,
#返回token_type_ids
return_token_type_ids=True,
#返回attention_mask
return_attention_mask=True,
#返回special_tokens_mask 特殊符号标识
return_special_tokens_mask=True,
#返回length 标识长度
return_length=True,
)
#input_ids 编码后的词
#token_type_ids 第1个句子和特殊符号的位置是0，第2个句子的位置是1
#special_tokens_mask 特殊符号的位置是1，其他位置是0
#attention_mask PAD的位置是0，其他位置是1
#length 返回句子长度
for k, v in out.items():
    print(k, ':', v)
print(tokenizer.decode(out['input_ids']))
###########################################################
#批量编码成对的句子

out = tokenizer.batch_encode_plus(
#编码成对的句子
batch_text_or_text_pairs=sents,
#batch_text_or_text_pairs=[sents[0], sents[1]],
add_special_tokens=True,
#当句子长度大于max_length时截断
truncation=True,
#一律补零，直到max_length长度
padding='max_length',
max_length=25,
#可取值tf、pt、np，默认为返回list
return_tensors=None,
#返回token_type_ids
return_token_type_ids=True,
#返回attention_mask
return_attention_mask=True,
#返回special_tokens_mask 特殊符号标识
return_special_tokens_mask=True,
#返回offsets_mapping 标识每个词的起止位置，这个参数只能BertTokenizerFast使用
#return_offsets_mapping=True,
#返回length 标识长度
return_length=True,
)
#input_ids 编码后的词
#token_type_ids 第1个句子和特殊符号的位置是0，第2个句子的位置是1
#special_tokens_mask 特殊符号的位置是1，其他位置是0
#attention_mask PAD的位置是0，其他位置是1
#length 返回句子长度
for k, v in out.items():
    print(k, ':', v)
print('================')
for x in out['input_ids']:
    print(tokenizer.decode(x))

###################################
#获取字典
vocab = tokenizer.get_vocab()
print(type(vocab), len(vocab), '明月' in vocab)



#第3章/加载数据集
from datasets import load_dataset
dataset = load_dataset("wangrui6/Zhihu-KOL")
print(dataset)