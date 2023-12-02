import torch
import torch.nn as nn
import numpy as np

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
        # 调整输入数据维度以匹配模型
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        # 调整维度以适应Transformer的输入
        x = x.permute(1, 0, 2)
        
        # 编码器
        encoder_output = self.transformer_encoder(x)
        
        # 解码器
        decoder_output = self.transformer_decoder(x, encoder_output)
        # 恢复原始维度
        decoder_output = decoder_output.permute(1, 0, 2)
        # 仅使用最后一个时间步的输出
        output = self.fc(decoder_output[:, -1, :])
        return output

# 读取数据
def get_data():
    file_path = "dataset.txt"
    with open(file_path, 'r') as file:
        file_content = file.read()
        lines = file_content.strip().split('\n')
        arrays = []
        for line in lines:
            values = line.split()
            float_values = [float(val) for val in values]
            arrays.append(np.array(float_values))
        big_array = np.array(arrays)
        return big_array.astype(np.float32)
    

# 模型和训练参数
input_size = 1
output_size = 1
model = TransformerModel(input_size, output_size)
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# 归一化数据
def normalize_data(data):
    return (data - np.mean(data)) / np.std(data)


# 处理数据
stock_data = get_data()
normalized_data = normalize_data(stock_data)
tensor_stock_data = torch.from_numpy(normalized_data)

# 训练模型
num_epochs = 300
target_data = tensor_stock_data[:, 1:]
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(tensor_stock_data[:, :-1])
    loss = criterion(output, target_data)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 预测最后一组数据的最后一个值
with torch.no_grad():
    test_input = tensor_stock_data[-1, :-1].unsqueeze(0)
    predicted = model(test_input)
    predicted_denormalized = predicted.item() * np.std(stock_data) + np.mean(stock_data)
    actual_value = stock_data[-1, -1]
    print(f"Predicted: {predicted_denormalized}, Actual: {actual_value}")
    
