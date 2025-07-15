import sys
import os

# Obtém o caminho absoluto para o diretório pai de 'src'
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Adiciona o diretório raiz ao sys.path se ele ainda não estiver lá
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm import tqdm
from src.dataset import NerDataset
from src.model import BioBERTptNER
from src.utils import load_data, tags_mapping_v2, tags_2_labels
from src.metrics import MetricsTracking
import time
import json
from transformers import get_linear_schedule_with_warmup

"""

# Load the data
train_df = load_data("data/split_0/df_test_tokens_labeled_iob_bert_format.csv")
dev_df = load_data("data/split_0/df_test_tokens_labeled_iob_bert_format.csv")
df_total = load_data("data/df_tokens_labeled_iob_bert_format.csv")

# Build Vocabulary
tag2idx, idx2tag , unseen_label, unique_tags = tags_mapping_v2(df_total["tags"])
train_df["labels"] = train_df["tags"].apply(lambda tags : tags_2_labels(unseen_label ,tags, tag2idx))
dev_df["labels"] = dev_df["tags"].apply(lambda tags : tags_2_labels(unseen_label ,tags, tag2idx))

# Build NerDataset
train_dataset = NerDataset(train_df, tag2idx)
dev_dataset = NerDataset(dev_df, tag2idx) 

# Model
model = BioBERTptNER(len(unique_tags))  # Adjust label count
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
#lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * 10)

# Hyperparameters
batch_size = [4, 8, 16]
epochs = 3
"""

j = 0
batch_size = 4
epochs = 10
lr = 3e-5
warmup_ratio = 0.1 

data_folder = "biobertpt/data/"
model_folder = "biobertpt/models/"
metrics_folder = "biobertpt/data/metrics/"

# Load the data
train_df = load_data(data_folder+"df_train_tokens_labeled_iob_bert_format.csv")
dev_df = load_data(data_folder+"df_test_tokens_labeled_iob_bert_format.csv")
df_total = load_data(data_folder+"df_tokens_labeled_iob_bert_format.csv")

# Build Vocabulary
tag2idx, idx2tag , unseen_label, unique_tags = tags_mapping_v2(df_total["tags"])
train_df["labels"] = train_df["tags"].apply(lambda tags : tags_2_labels(unseen_label ,tags, tag2idx))
dev_df["labels"] = dev_df["tags"].apply(lambda tags : tags_2_labels(unseen_label ,tags, tag2idx))

# Build NerDataset
train_dataset = NerDataset(train_df, tag2idx)
dev_dataset = NerDataset(dev_df, tag2idx) 

# Model
model = BioBERTptNER(len(unique_tags))  # Adjust label count
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

# Training Loop
start = time.time()
train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
dev_dataloader = DataLoader(dev_dataset, batch_size = batch_size, shuffle = True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Calculate total training steps and warmup steps
total_steps = len(train_dataloader) * epochs
warmup_steps = int(total_steps * warmup_ratio)

# Create the learning rate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)
for epoch in range(epochs):

    train_metrics = MetricsTracking()
    total_loss_train = 0

    model.train() #train mode

    for train_data, train_label in tqdm(train_dataloader):
        train_label = train_label.to(device)
        '''
        squeeze in order to match the sizes. From [batch,1,seq_len] --> [batch,seq_len]
        '''
        mask = train_data['attention_mask'].squeeze(1).to(device)
        input_id = train_data['input_ids'].squeeze(1).to(device)

        optimizer.zero_grad()

        output = model(input_id, mask, train_label)
        loss, logits = output.loss, output.logits
        predictions = logits.argmax(dim= -1)

        #compute metrics
        train_metrics.update(predictions, train_label)
        total_loss_train += loss.item()

        #grad step
        loss.backward()
        optimizer.step()
        lr_scheduler.step()  # Update the learning rate
    
    train_results = train_metrics.return_avg_metrics(len(train_dataloader))
    print(f"EPOCH: {epoch}\nModel : LR={lr}_Batch_Size={batch_size}\nTRAIN \nLoss: {total_loss_train / len(train_dataset)} \nMetrics {train_results}\n" )

end = time.time()
fine_tuning_time = end - start
print(f"Fine-tuning Time: {fine_tuning_time} seconds\n")

'''
EVALUATION MODE
'''
model.eval()

dev_metrics = MetricsTracking()
total_loss_dev = 0

with torch.no_grad():
    for dev_data, dev_label in tqdm(dev_dataloader):

        dev_label = dev_label.to(device)

        mask = dev_data['attention_mask'].squeeze(1).to(device)
        input_id = dev_data['input_ids'].squeeze(1).to(device)

        output = model(input_id, mask, dev_label)
        loss, logits = output.loss, output.logits

        predictions = logits.argmax(dim= -1)

        dev_metrics.update(predictions, dev_label)
        total_loss_dev += loss.item()

    dev_results = dev_metrics.return_avg_metrics(len(dev_dataloader))
    train_results['fine_tuning_time'] = fine_tuning_time

    metric_file_name = metrics_folder + f'classification_report_lr={lr}_batch_size={batch_size}.json'

    with open(metric_file_name, "w") as f:
        json.dump(dev_results, f, indent=4)

    print(f"VALIDATION \nLoss {total_loss_dev / len(dev_dataset)} \nMetrics{dev_results}\n" )

    model_file_name = model_folder+f"model_lr={lr}_batch_size={batch_size}.pth"
    torch.save(model.state_dict(), model_file_name)
        