import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report

# 假设你有一个包含验证集的数据集
validation_texts = [
"施工现场未经充分质量检测和监测机制支持，可能导致施工过程中的质量问题难以及时发现，对整体工程质量构成潜在威胁。",

"质量监测体系缺失可能使得施工中的细微质量问题未被及时追踪，进而影响工程的整体质量表现。",

"缺乏明确的质量检测标准和程序可能导致施工现场存在潜在的质量风险，对工程整体质量产生潜在不利影响。",

"施工团队未建立健全的质量监测机制，可能导致工程过程中的质量问题无法及时发现和纠正，对最终工程质量构成风险。",

"在工程项目中缺乏明确的质量控制流程可能使得施工过程中的质量问题积聚难以察觉，对工程总体质量带来潜在威胁。",

"施工现场缺乏适当的紧急疏散演练和定期的安全演练，可能导致在紧急情况下工人无法迅速响应，对施工安全带来潜在风险。",

"未进行定期的安全演练和紧急疏散演练可能使得工人在危急时刻难以做出迅速而正确的反应，对施工现场的整体安全构成潜在威胁。",

"施工团队未定期进行紧急疏散演练，可能导致工人在紧急情况下面临混乱和不确定性，对施工安全形成潜在危险。",

"缺乏定期的安全培训和演练可能使得工人在突发情况下缺乏足够的准备，对施工现场的整体安全性构成潜在风险。",

"施工现场未建立适当的安全演练机制，可能使工人在紧急状况下无法迅速有效地采取安全措施，对施工安全造成潜在的威胁。",

"极端天气条件，如极端温度或持续降雨，可能引起工程进度的不稳定性。",

"在极端天气条件下，例如极端高温或连续降雨，可能导致工程进度不稳定。",

"工程可能面临不稳定的进度，当遭遇恶劣天气条件，如异常高温或长时间降雨时。",

"持续的极端天气，如极端高温或不断降雨，有可能对工程进度造成不稳定的影响。",

"不利的天气条件，如持续的高温或连续降雨，可能导致工程进度的不稳定性。",

"工程管理得当，现场质量层层把关。",

"工程管理实践卓越，现场质量得以稳定维护。"
# 验证集文本
]
print("validation_texts input: ", validation_texts)
validation_labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]  # 对应的标签（风险类型）

# 数据预处理，创建验证集数据集
class ValidationDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label)
        }

# 创建验证集数据集和数据加载器#
#validation_texts = np.pad(validation_texts, (0, 6-validation_texts.shape[0]), 'constant', constant_values=(0, -1))
#validation_texts = np.pad(validation_texts, (0, 80), 'constant', constant_values=0)

validation_dataset = ValidationDataset(validation_texts, validation_labels)
validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

# 加载模型
model = BertForSequenceClassification.from_pretrained('path/to/save/model')

# 评估模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in validation_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # 模型推断
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # 获取预测结果
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

with torch.no_grad():
    for batch in validation_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # 模型推断
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # 获取预测结果
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

# 打印文本数据和对应的模型预测结果
for text, label, pred in zip(validation_texts, all_labels, all_preds):
    print(f"文本: {text}")
    print(f"真实标签: {label}, 模型预测: {pred}")
    print("-" * 50)
# 计算准确性
accuracy = accuracy_score(all_labels, all_preds)
print(f"准确性：{accuracy}")

# 计算其他性能指标
print(classification_report(all_labels, all_preds))
