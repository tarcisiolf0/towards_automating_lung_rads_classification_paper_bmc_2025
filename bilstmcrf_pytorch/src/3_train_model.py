import sys
import os

# Obtém o caminho absoluto para o diretório pai de 'src'
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Adiciona o diretório raiz ao sys.path se ele ainda não estiver lá
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import torch
import torch.nn as nn
import torch.optim as optim
from torchcrf import CRF
import utils.data_processing as dp
from seqeval.metrics import classification_report
import itertools
import json
import logging
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class Config:
    def __init__(self, vocab_size, num_tags, embedding_dim=128, hidden_dim=64, lstm_dropout=0.1, learning_rate=0.01, epochs=10, batch_size=2, padding_idx=-1):
        self.vocab_size = vocab_size
        self.num_tags = num_tags
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lstm_dropout = lstm_dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.padding_idx = padding_idx

class BiLSTM_CRF(nn.Module):
    def __init__(self, config, word2index):
        super(BiLSTM_CRF, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=word2index['<PAD>'])
        self.lstm = nn.LSTM(config.embedding_dim, config.hidden_dim // 2, num_layers=2, bidirectional=True, batch_first=True, dropout=config.lstm_dropout)
        self.hidden2tag = nn.Linear(config.hidden_dim, config.num_tags)
        self.crf = CRF(config.num_tags, batch_first=True)

    def forward(self, sentence):
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds)
        emissions = self.hidden2tag(lstm_out)
        return emissions


def train(model, train_sentences_tensor, train_tags_tensor, config, word2index):
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    dataset = TensorDataset(train_sentences_tensor, train_tags_tensor)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        batch_num = 0

        for batch_sentences, batch_tags in dataloader:

            batch_num += 1

            model.zero_grad()
            emissions = model(batch_sentences)
            mask = batch_sentences != word2index['<PAD>']
            loss = -model.crf(emissions, batch_tags, mask=mask)

            loss.backward() 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()

            avg_batch_loss = loss.item()
            logging.info(f"Epoch: {epoch+1}, Batch: {batch_num}, Loss: {avg_batch_loss:.4f}")

        avg_epoch_loss = total_loss / len(dataloader)
        logging.info(f"Epoch: {epoch+1}, Average Loss: {avg_epoch_loss:.4f}")

    return model

def predict(model, sentence, word2index):
    model.eval()
    with torch.no_grad():
        #sentence = sentence.to(device)
        emissions = model(sentence)
        mask = sentence != word2index['<PAD>']
        predicted_tags = model.crf.decode(emissions, mask=mask)
    return predicted_tags[0]

def evaluate_model(model, test_sentences_tensor, test_tags_tensor, test_data, word2index, tag2index, index2tag):
    test_actual_tags = []
    test_predicted_tags = []

    for i in range(len(test_sentences_tensor)):
        _, tags = dp.sentence_to_indices(test_data[i]["sentence"], word2index, tag2index) 
        predicted_tags = predict(model, test_sentences_tensor[i].unsqueeze(0), word2index)

        actual_tags = test_tags_tensor[i].tolist()
        actual_tags = [index2tag[idx] for idx in actual_tags]

        predicted_tags = [index2tag[idx] for idx in predicted_tags]

        actual_tags = actual_tags[:len(tags)]
        predicted_tags = predicted_tags[:len(tags)]

        test_actual_tags.append(actual_tags)
        test_predicted_tags.append(predicted_tags)

    return test_actual_tags, test_predicted_tags


def convert(o):
    if isinstance(o, np.integer):
        return int(o)
    elif isinstance(o, np.floating):
        return float(o)
    elif isinstance(o, np.ndarray):
        return o.tolist()
    else:
        return o
    
if __name__ == "__main__":

    folder = "bilstmcrf_pytorch/data/"
    metrics_folder = "bilstmcrf_pytorch/data/metrics/"
    models_folder = "bilstmcrf_pytorch/models/"

    train_data = dp.load_json(folder+"train_data.json")
    test_data = dp.load_json(folder+"test_data.json")
    unique_tags = dp.load_json(folder+"unique_tags.json")
    word2index = dp.load_json(folder+"word2index.json")
    index2word = {int(k): v for k, v in dp.load_json(folder+"index2word.json").items()}
    tag2index = dp.load_json(folder+"tag2index.json")
    index2tag = {int(k): v for k, v in dp.load_json(folder+"index2tag.json").items()}


    max_len = 512
    train_sentences_indices, train_tags_indices = dp.process_data(train_data, max_len, word2index, tag2index)
    test_sentences_indices, test_tags_indices = dp.process_data(test_data, max_len, word2index, tag2index)

    vocab_size = len(word2index)
    num_tags = len(unique_tags)
    best_f1_score = -1

    param_grid = {
        "batch_size": [4],
        "embedding_dim": [50],
        "hidden_dim": [50],
        "lstm_dropout": [0.1],
        "learning_rate": [0.01],
    }

    epochs = 10
    best_f1_score = -1

    for batch_size, embedding_dim, hidden_dim, lstm_dropout, learning_rate in itertools.product(
        param_grid["batch_size"], param_grid["embedding_dim"], param_grid["hidden_dim"], param_grid["lstm_dropout"], param_grid["learning_rate"]):

        start = time.time()
        logging.info(f"Training with batch_size={batch_size}, embedding_dim={embedding_dim}, hidden_dim={hidden_dim}_lstm_dropout={lstm_dropout}_learning_rate={learning_rate}\n")

        config = Config(vocab_size, num_tags, embedding_dim, hidden_dim, lstm_dropout, learning_rate, epochs=epochs, batch_size=batch_size, padding_idx=word2index['<PAD>'])
        model = BiLSTM_CRF(config, word2index)
        trained_model = train(model, train_sentences_indices, train_tags_indices, config, word2index)
        
        end = time.time()
        training_time = end - start

        logging.info(f"Training time: {end - start:.4f} seconds\n")
        test_actual_tags, test_predicted_tags = evaluate_model(trained_model, test_sentences_indices, test_tags_indices, test_data, word2index, tag2index, index2tag)

        report = classification_report(test_actual_tags, test_predicted_tags, output_dict=True)
        report['training_time'] = training_time  # Add training time to the report

        f1_score = report["macro avg"]["f1-score"]
        logging.info(f"F1-score: {f1_score:.4f}\n")

        metric_file_name = metrics_folder + f"best_model_classification_report_batch_size={batch_size}_embedding_dim={embedding_dim}_hidden_dim={hidden_dim}_lstm_dropout={lstm_dropout}_learning_rate={learning_rate}.json"
        
        #dp.save_json(report, metric_file_name)

        # Save report as a JSON file
        with open(metric_file_name, "w") as f:
            json.dump(report, f, indent=4, default=convert)

        model_file_name = models_folder + f"best_model_bilstm_crf_batch_size={batch_size}_embedding_dim={embedding_dim}_hidden_dim={hidden_dim}_lstm_dropout={lstm_dropout}_learning_rate={learning_rate}.pth"

        if f1_score > best_f1_score:
            best_f1_score = f1_score
            best_params = {"batch_size": batch_size, "embedding_dim": embedding_dim, "hidden_dim": hidden_dim}

        torch.save(trained_model.state_dict(), model_file_name)

        logging.info(f"Best Model - batch_size={best_params['batch_size']}, embedding_dim={best_params['embedding_dim']}, hidden_dim={best_params['hidden_dim']}, F1-score={best_f1_score:.4f}")
        logging.info(f"Metric_file_name {metric_file_name}\n")
        logging.info(f"Model_file_name {model_file_name}\n")