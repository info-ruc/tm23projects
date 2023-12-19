import torch
from tqdm import tqdm
import torch.nn as nn
from datasets import load_from_disk
from transformers import BertTokenizer,BertConfig,AdamW,BertModel
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 定义神经网络
class BertClassificationModel(nn.Module):
	def __init__(self):
		super(BertClassificationModel, self).__init__()   
		#加载预训练模型
		pretrained_weights=r"E:\model\bert-base-chinese"
        #定义Bert模型
		self.bert = BertModel.from_pretrained(pretrained_weights)
		for param in self.bert.parameters():
			param.requires_grad = True
		#定义线性函数      
		self.dense = nn.Linear(768, 2)  #bert默认的隐藏单元数是768， 输出单元是2，表示二分类
	def forward(self, input_ids,token_type_ids,attention_mask):
		#得到bert_output
		bert_output = self.bert(input_ids=input_ids,token_type_ids=token_type_ids, attention_mask=attention_mask)
		#获得预训练模型的输出
		bert_cls_hidden_state = bert_output[1]
		#将768维的向量输入到线性层映射为二维向量
		linear_output = self.dense(bert_cls_hidden_state)
		return  linear_output

# 使用BertTokenizer 编码成Bert需要的输入格式
def encoder(max_len,vocab_path,text_list):
	#将text_list embedding成bert模型可用的输入形式
	#加载分词模型
	tokenizer = BertTokenizer.from_pretrained(vocab_path)
	tokenizer = tokenizer(
		text_list,
		padding = True,
		truncation = True,
		max_length = max_len,
		return_tensors='pt'  # 返回的类型为pytorch tensor
		)
	input_ids = tokenizer['input_ids']
	token_type_ids = tokenizer['token_type_ids']
	attention_mask = tokenizer['attention_mask']
	return input_ids,token_type_ids,attention_mask

# 将数据加载为Tensor格式
def load_data(Dataset):
	text_list = []
	labels = []
	for item in Dataset:
		#label在什么位置就改成对应的index
		label = int(item['label'])
		text = item['text']
		text_list.append(text)
		labels.append(label)
# 调用encoder函数，获得预训练模型的三种输入形式
	input_ids,token_type_ids,attention_mask = encoder(max_len=150,vocab_path=r"E:\model\bert-base-chinese\vocab.txt",text_list=text_list)
	labels = torch.tensor(labels)
	#将encoder的返回值以及label封装为Tensor的形式
	data = TensorDataset(input_ids,token_type_ids,attention_mask,labels)
	return data

#实例化DataLoader
#设定batch_size
batch_size = 16
#从磁盘加载数据
dataset = load_from_disk('E:\datasets\ChnSentiCorp')
#取出训练集
dataset_train = dataset['train']
dataset_validation = dataset['validation']
#调用load_data函数，将数据加载为Tensor形式
dataset_train_ts = load_data(dataset_train)
dataset_validation_ts = load_data(dataset_validation)
#将训练数据和测试数据进行DataLoader实例化
train_loader = DataLoader(dataset=dataset_train_ts, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(dataset=dataset_validation_ts, batch_size=batch_size, shuffle=True)

# 定义验证函数
def dev(model,validation_loader):
	#将模型放到服务器上
	model.to(device)
	#设定模式为验证模式
	model.eval()
	#设定不会有梯度的改变仅作验证
	with torch.no_grad():
		correct = 0
		total = 0
		for step, (input_ids,token_type_ids,attention_mask,labels) in tqdm(enumerate(validation_loader),desc='Dev Itreation:'):
			input_ids,token_type_ids,attention_mask,labels=input_ids.to(device),token_type_ids.to(device),attention_mask.to(device),labels.to(device)
			out_put = model(input_ids,token_type_ids,attention_mask)
			_, predict = torch.max(out_put.data, 1)
			correct += (predict==labels).sum().item()
			total += labels.size(0)
		res = correct / total
		return res

# 定义训练函数 
def train(model,train_loader,validation_loader):
	#将model放到服务器上
	model.to(device)
	#设定模型的模式为训练模式
	model.train()
	#定义模型的损失函数
	criterion = nn.CrossEntropyLoss()
	param_optimizer = list(model.named_parameters())
	no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
	#设置模型参数的权重衰减
	optimizer_grouped_parameters = [
		{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
		'weight_decay': 0.01},
		{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
	]
	#学习率的设置
	optimizer_params = {'lr': 1e-5, 'eps': 1e-6, 'correct_bias': False}
	#使用AdamW 主流优化器
	optimizer = AdamW(optimizer_grouped_parameters, **optimizer_params)
	#学习率调整器，检测准确率的状态，然后衰减学习率
	scheduler = ReduceLROnPlateau(optimizer,mode='max',factor=0.5,min_lr=1e-7, patience=5,verbose= True, threshold=0.0001, eps=1e-08)
	t_total = len(train_loader)
	#设定训练轮次
	total_epochs = 2
	bestAcc = 0
	correct = 0
	total = 0
	print('Training and verification begin!')
	for epoch in range(total_epochs): 
		for step, (input_ids,token_type_ids,attention_mask,labels) in enumerate(train_loader):
			#从实例化的DataLoader中取出数据，并通过 .to(device)将数据部署到服务器上    input_ids,token_type_ids,attention_mask,labels=input_ids.to(device),token_type_ids.to(device),attention_mask.to(device),labels.to(device)
			#梯度清零
			optimizer.zero_grad()
			#将数据输入到模型中获得输出
			out_put =  model(input_ids,token_type_ids,attention_mask)
			#计算损失
			loss = criterion(out_put, labels)
			_, predict = torch.max(out_put.data, 1)
			correct += (predict == labels).sum().item()
			total += labels.size(0)
			loss.backward()
			optimizer.step()
			#每两步进行一次打印
			if (step + 1) % 2 == 0:
				train_acc = correct / total
				print("Train Epoch[{}/{}],step[{}/{}],tra_acc{:.6f} %,loss:{:.6f}".format(epoch + 1, total_epochs, step + 1, len(train_loader),train_acc*100,loss.item()))
			#每五十次进行一次验证
			if (step + 1) % 50 == 0:
				train_acc = correct / total
				#调用验证函数dev对模型进行验证，并将有效果提升的模型进行保存
				acc = dev(model, validation_loader)
				if bestAcc < acc:
					bestAcc = acc
					#模型保存路径
					path = r"E:\output\savedmodel\model_new.pkl"
					torch.save(model, path)
				print("DEV Epoch[{}/{}],step[{}/{}],tra_acc:{:.6f} %,bestAcc{:.6f}%,dev_acc{:.6f} %,loss:{:.6f}".format(epoch + 1, total_epochs, step + 1, len(train_loader),train_acc*100,bestAcc*100,acc*100,loss.item()))
		scheduler.step(bestAcc)

# 设备配置
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
#实例化模型
model = BertClassificationModel()
#调用训练函数进行训练与验证
train(model,train_loader,validation_loader)

# 定义预测函数
import torch.nn.functional as F
def predict(model,test_loader):
    model.to(device)
    # 将模型中的某些特定层或部分切换到评估模式
    model.eval()
    predicts = []
    predict_probs = []
    with torch.no_grad():
        correct = 0
        total = 0
        for step, (input_ids,token_type_ids,attention_mask,labels) in enumerate(test_loader): 
            input_ids,token_type_ids,attention_mask,labels=input_ids.to(device),token_type_ids.to(device),attention_mask.to(device),labels.to(device)
            out_put = model(input_ids,token_type_ids,attention_mask)
            _, predict = torch.max(out_put.data, 1)
            pre_numpy = predict.cpu().numpy().tolist()
            print("pre_numpy:")
            print(pre_numpy)
            print("labels:")
            print(labels)
            predicts.extend(pre_numpy)
            probs = F.softmax(out_put, dim=1).detach().cpu().numpy().tolist()
            predict_probs.extend(probs)
            correct += (predict==labels).sum().item()
            total += labels.size(0)
        res = correct / total
        print('**************结果**************\npredict_Accuracy : {} %'.format(100 * res))
        #返回预测结果和预测的概率
        return predicts,predict_probs

# 使用训练好的模型进行预测
# 1、加载测试数据集
dataset_test = dataset['test']
dataset_test_ts = load_data(dataset_test)
test_loader = DataLoader(dataset=dataset_test_ts, batch_size=batch_size, shuffle=False) 
# 2、加载训练好的模型
path = r'E:\output\savedmodel\model_new.pkl'
Trained_model = torch.load(path)
# 3、开始预测
print("The prediction start !\n**************说明**************\n一、1代表正向情感，0代表负面情感。\n二、预测值代表“训练好的模型对该句子所预测的情感”，对应变量pre_numpy。\n三、真实值代表“实际上该句子所表达的情感”，对应变量labels。\n********************************")
#predicts是预测的（0或1），predict_probs是概率值
predicts,predict_probs = predict(Trained_model,test_loader)
#predicts
#predict_probs

