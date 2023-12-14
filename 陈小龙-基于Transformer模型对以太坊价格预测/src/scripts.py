import torch
import torch.nn as nn
import numpy as np
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# 创建Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256, num_layers=6, num_heads=8, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        
        # 编码器层
        self.encoder_layer = nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size, dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        
        # 解码器层
        self.decoder_layer = nn.TransformerDecoderLayer(hidden_size, num_heads, hidden_size, dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers)
        
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 确保输入数据的形状为 [sequence_length, batch_size, input_size]
        x = x.unsqueeze(0)
        
        # 将 x 转换为 [64, 1, 3]
        x = x.permute(1, 0, 2)
        
        # 修改特征维度为 256
        linear_layer = nn.Linear(3, 256)

        # 使用GPU计算
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        linear_layer = linear_layer.to(device)
        x = x.to(device)

        # 将数据重塑为 [64, 3]，为线性层做准备
        x = x.view(-1, 3)
        
        # 线性变换，将特征维度从 3 变为 256
        x = linear_layer(x)

        # 假设线性层后的形状为 [batch_size, 256]，根据实际的批量大小进行重塑, 动态获取批量大小
        batch_size = x.size(0)
        
        # 动态重塑，以匹配模型期望的输入形状
        x = x.view(batch_size, 1, -1)
        
        # 编码器
        encoder_output = self.transformer_encoder(x)
        
        # 解码器
        decoder_output = self.transformer_decoder(x, encoder_output)
        
        # 恢复原始维度
        decoder_output = decoder_output.permute(1, 0, 2)
        
        # 保留批处理输出
        output = self.fc(decoder_output)
        
        return output



# 读取数据
def get_data():
    file_path = "dataset.txt"
    with open(file_path, 'r') as file:
        lines = file.read().strip().split('\n')[1:]

        input_data = []
        input_row = []

        for idx, line in enumerate(lines, start=1):
            values = line.split()
            input_row.append(float(values[1]))
            input_row.append(float(values[2]))
            input_row.append(float(values[3]))

            input_data.append(input_row)
            input_row = []

        # 将二维数组转换为 PyTorch 的张量
        return np.array(input_data, dtype=np.float32)


# 模型和训练参数
input_size = 3
output_size = 3

model = TransformerModel(input_size, output_size)

#criterion = nn.MSELoss()
criterion = nn.SmoothL1Loss()

#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


# 归一化数据
def normalize_data(data):
    return (data - np.mean(data)) / np.std(data)


# 处理数据
data = get_data()
tensor_data = torch.from_numpy(normalize_data(data))


# 使用GPU计算
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
tensor_data = tensor_data.to(device)

# 将数据转换为PyTorch数据集
dataset = TensorDataset(tensor_data[:-1],tensor_data[1:])
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # 逐步构建输入序列和目标序列并进行训练
    for train_data, target_data in dataloader:
        train_data, target_data = train_data.to(device),  target_data.to(device)
        output = model(train_data)
        # 将目标数据的形状扩展为 [1, 64, 3]
        target_data = target_data.unsqueeze(0)
        loss = criterion(output, target_data)
        # 给第三个参数一个较大的权重
        weights = torch.tensor([1.0, 1.0, 1.2], device=device)
        # 以平方误差为例
        loss_weighted = torch.mean(weights * (loss ** 2))
        loss_weighted.backward()


    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


# 预测最后的64个值并与实际值比较
with torch.no_grad():
    # 选择最后的64个值
    test_input = tensor_data[-64:].to(device)
    # 进行预测
    predicted = model(test_input)
    # 重塑以分离每个单独的预测
    predicted_values = predicted.cpu().numpy().reshape(-1, output_size)
    # 反归一化预测值
    predicted_denormalized = predicted_values * np.std(data, axis=0) + np.mean(data, axis=0)
    actual_values = data[-64:]

    # 输出预测值和实际值以便比较
    for i in range(64):
        print(f"预测值: {predicted_denormalized[i]}, 实际值: {actual_values[i]}")

    #选择价格的预测值和实际值
    predicted_prices = predicted_denormalized[:, 2]  # 第三列是索引为2的列
    actual_prices = actual_values[:, 2]  # 第三列是索引为2的列

    # 创建一个新的图表
    plt.figure(figsize=(10, 6))

    # 绘制预测值和实际值的对比图
    plt.plot(predicted_prices, label="Predict Price")
    plt.plot(actual_prices, label="Actual Price")

    # 添加图例、标题和标签
    plt.legend()
    plt.title("Ethereum Price Prediction Upon Transformer Model")
    plt.xlabel("Day")
    plt.ylabel("Price")

    # 设置Y轴值范围
    plt.ylim(1000, 3000)

    # 显示图表
    plt.show()
    
