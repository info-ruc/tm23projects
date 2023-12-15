import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# 加载已经训练好的模型和tokenizer
model = BertForSequenceClassification.from_pretrained('bert_sentiment_model')
tokenizer = BertTokenizer.from_pretrained('bert_sentiment_model')

# 加载其他验证数据集
# 替换 'other_val_texts' 和 'other_val_labels' 为你的实际数据集
other_val_texts = [
    "项目管理良好，进度质量都正常",

    "工程项目管理注重信息化建设，采用先进的技术手段，提高项目管理的效率和精准度。",

    "项目经理具备团队领导能力，激发团队成员的潜力，确保项目团队的高效协同。",

    "工程管理层注重沟通技巧，与各利益相关方保持良好沟通，提高项目的透明度和合作度。",

    "项目管理层关注法规合规，确保项目在合法合规的基础上进行，降低法律风险。",

    "工程项目管理注重可行性研究，在项目初期进行详尽的可行性分析，降低项目失败的风险。",

    "项目变更可能需要额外的审批和文件工作，导致计划调整，对进度产生不利影响。",

    "施工现场的土壤条件可能与预期不符，可能需要重新评估基础设计，对工程进度产生延误。",

    "全球性的大流行病（如疫情）可能导致劳工不足、物资供应不稳定，进而影响工程进度。",

    "施工队伍的培训水平不足可能导致施工过程中出现错误，增加修改和重新工作的需求，对进度产生负面影响。",

    "现场安全事故可能导致工程现场暂时关闭，需要额外时间来进行调查和修复，对进度形成打击。",
]  # 其他验证数据集文本
other_val_labels = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,]  # 其他验证数据集标签

# 创建数据集和数据加载器
class CustomDataset:
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

other_val_dataset = CustomDataset(other_val_texts, other_val_labels, tokenizer, max_len=128)
other_val_loader = DataLoader(other_val_dataset, batch_size=8, shuffle=False)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 验证模型
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for i, batch in enumerate(tqdm(other_val_loader, desc='Other Validation')):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        # 打印每一条数据的预测值和实际值
        for j in range(len(preds)):
            print(f"Example {i * 8 + j + 1}:")
            print(f"Text: {other_val_texts[i * 8 + j]}")
            print(f"Actual Label: {other_val_labels[i * 8 + j]}")
            print(f"Predicted Label: {preds[j]}")
            print("="*30)

# 计算准确性
accuracy = accuracy_score(all_labels, all_preds)

print(f'Other Validation Accuracy: {accuracy:.4f}')