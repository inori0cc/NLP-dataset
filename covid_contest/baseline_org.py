import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW





train_df = pd.read_csv('train.csv', sep='\t')
test_df = pd.read_csv('test.csv', sep='\t')

# 分词器，词典
tokenizer= BertTokenizer.from_pretrained('bert-base-chinese')
train_encoding = tokenizer(train_df['text'].tolist(), truncation=True, padding=True, max_length=128)
test_encoding = tokenizer(test_df['text'].tolist(), truncation=True, padding=True, max_length=128)



# 数据集读取
class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    # 读取单个样本
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item
    
    def __len__(self):
        return len(self.labels)

train_dataset = NewsDataset(train_encoding, train_df['label'])
test_dataset = NewsDataset(test_encoding, [0] * len(test_df))




# 精度计算
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)



model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 单个读取到批量读取
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 优化方法
optim = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * 1


# 训练函数
def train():
    model.train()
    total_train_loss = 0
    iter_num = 0
    total_iter = len(train_loader)
    for batch in train_loader:
        # 正向传播
        optim.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        total_train_loss += loss.item()
        
        # 反向梯度信息
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # 参数更新
        optim.step()

        iter_num += 1
        if(iter_num % 100==0):
            print("epoth: %d, iter_num: %d, loss: %.4f, %.2f%%" % (epoch, iter_num, loss.item(), iter_num/total_iter*100))
        
    print("Epoch: %d, Average training loss: %.4f"%(epoch, total_train_loss/len(train_loader)))

for epoch in range(1):
    print("------------Epoch: %d ----------------" % epoch)
    train()



with torch.no_grad():
    pred_label = []
    for batch in test_dataloader:
        # 正向传播
        optim.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        pred_label += list(outputs.logits.argmax(1).cpu().data.numpy())




pd.DataFrame(
    {
        'label': pred_label
    }
).to_csv('submit.csv', index=None)

