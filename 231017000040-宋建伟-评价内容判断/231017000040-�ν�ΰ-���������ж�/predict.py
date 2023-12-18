import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import XLNetTokenizer, XLNetModel
from transformers import AutoTokenizer

# 检查是否有可用的GPU设备
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class XLNetCNN(nn.Module):
    def __init__(self, n_filters, filter_sizes, output_dim, dropout):
        super(XLNetCNN, self).__init__()
        self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased')
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n_filters, (fs, self.xlnet.config.hidden_size)) for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ids, mask):
        x = self.xlnet(ids, attention_mask=mask)[0]
        x = x.unsqueeze(1)
        x = [nn.functional.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in x]
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# 加载模型和tokenizer，并将其移动到GPU上
output_dim = 2  # 分类数目
n_filters = 100  # 滤波器的数量
filter_sizes = [2, 3, 4]  # 卷积核的大小
dropout = 0.5

# 实例化模型
model = XLNetCNN(n_filters, filter_sizes, output_dim, dropout)
model.load_state_dict(torch.load('model_weights_epoch10.pt'))
model.to(device)
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

# 输入一句话
input_text = input("请输入一句话：")

# 数据预处理
encoding = tokenizer.encode_plus(
    input_text,
    add_special_tokens=True,
    max_length=128,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)
input_ids = encoding['input_ids'].squeeze().to(device)
attention_mask = encoding['attention_mask'].squeeze().to(device)

# 模型预测
model.eval()
with torch.no_grad():
    output = model(input_ids.unsqueeze(0), attention_mask.unsqueeze(0))
    predicted_label = torch.argmax(output).item()

# 输出预测结果
print("预测结果：", predicted_label)