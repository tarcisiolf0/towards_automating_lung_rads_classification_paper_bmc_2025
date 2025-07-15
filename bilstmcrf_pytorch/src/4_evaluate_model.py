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
import pandas as pd

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

def predict(model, sentence, word2index):
    model.eval()
    with torch.no_grad():
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

def create_token_tag_dataframe(test_sentences_indices, test_actual_tags, test_predicted_tags, index2word):
    """
    Creates a pandas DataFrame with 'Token', 'Tag', and 'Predicted_Tag' columns.

    Args:
        test_sentences_indices: List of lists of token indices.
        test_actual_tags: List of lists of actual tags.
        test_predicted_tags: List of lists of predicted tags.
        index2word: Dictionary mapping token indices to words.

    Returns:
        pandas.DataFrame: DataFrame containing token, actual tag, and predicted tag information.
    """
    df = pd.DataFrame(columns=['token', 'iob_tag', 'predicted_iob_tag'])
    for test_sentence, test_actual_tag, test_predicted_tag in zip(test_sentences_indices, test_actual_tags, test_predicted_tags):
        test_sentence = test_sentence.tolist()
        test_sentence = test_sentence[:len(test_actual_tag)]
        test_sentence = [index2word[index] for index in test_sentence]

        for i in range(len(test_sentence)):
            data = []
            token = test_sentence[i]
            tag = test_actual_tag[i]
            predicted_tag = test_predicted_tag[i]
            data.append({'token': token, 'iob_tag': tag, 'predicted_iob_tag': predicted_tag})
            df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)

    return df

def create_test_csv_data_bilstm(csv_test_bert, df_tokens_labeled_iob, output_file_name):
    df_test_bert = pd.read_csv(csv_test_bert, encoding='utf-8')
    df_tokens_labeled_iob = pd.read_csv(df_tokens_labeled_iob, encoding='utf-8')

    test_index = df_test_bert['report'].unique().tolist() 
    test_index = [int(i) for i in test_index]

    filtered_df = df_tokens_labeled_iob[df_tokens_labeled_iob['report_index'].isin(test_index)]

    filtered_df.to_csv(output_file_name, index=False, encoding='utf-8')

if __name__ == "__main__":
    bert_test_csv_filename = "biobertpt/data/df_test_tokens_labeled_iob_bert_format.csv"
    df_tokens_labeled_iob = "bilstmcrf_pytorch/data/df_tokens_labeled_iob.csv"
    output_file_name = "bilstmcrf_pytorch/data/lung_rads/lung_rads_test.csv"
    create_test_csv_data_bilstm(bert_test_csv_filename, df_tokens_labeled_iob, output_file_name)

    test_data = dp.load_json("bilstmcrf_pytorch/data/test_data.json")
    unique_tags = dp.load_json("bilstmcrf_pytorch/data/unique_tags.json")
    word2index = dp.load_json("bilstmcrf_pytorch/data/word2index.json")
    index2word = {int(k): v for k, v in dp.load_json("bilstmcrf_pytorch/data/index2word.json").items()}
    tag2index = dp.load_json("bilstmcrf_pytorch/data/tag2index.json")
    index2tag = {int(k): v for k, v in dp.load_json("bilstmcrf_pytorch/data/index2tag.json").items()}

    output_file_name = "bilstmcrf_pytorch/data/lung_rads/lung_rads_test_predicted.csv"
    test_csv_file_name = "bilstmcrf_pytorch/data/lung_rads/lung_rads_test.csv"

    max_len = 512
    test_sentences_indices, test_tags_indices = dp.process_data(test_data, max_len, word2index, tag2index)

    vocab_size = len(word2index)
    num_tags = len(unique_tags)
    best_f1_score = -1

    embedding_dim = 50
    lstm_dropout = 0.1
    learning_rate = 0.01
    batch_size = 4
    hidden_dim = 50

    logging.info(f"Evaluating model")
    
    config = Config(vocab_size, num_tags, embedding_dim, hidden_dim, lstm_dropout, learning_rate, epochs=10, batch_size=batch_size, padding_idx=word2index['<PAD>'])

    model = BiLSTM_CRF(config, word2index)
    model_file_name = "bilstmcrf_pytorch/models/best_model_bilstm_crf_batch_size=4_embedding_dim=50_hidden_dim=50_lstm_dropout=0.1_learning_rate=0.01.pth"

    model.load_state_dict(torch.load(model_file_name))
    model.eval()

    test_actual_tags, test_predicted_tags = evaluate_model(model, test_sentences_indices, test_tags_indices, test_data, word2index, tag2index, index2tag)
    
    logging.info(f"Creating dataframe with results")
    
    df = create_token_tag_dataframe(test_sentences_indices, test_actual_tags, test_predicted_tags, index2word)
    test_df = pd.read_csv(test_csv_file_name)
    report_index_df = test_df.report_index
    df.insert(0, 'report_idx', report_index_df)
    df.to_csv(output_file_name, index=False)
  
    
    report = classification_report(test_actual_tags, test_predicted_tags, output_dict=True)

    f1_score = report["macro avg"]["f1-score"]
    logging.info(f"F1-score: {f1_score:.4f} ")

    output_file_name = f"bilstmcrf_pytorch/data/lung_rads/evaluation_report_batch_size={batch_size}_embedding_dim={embedding_dim}_hidden_dim={hidden_dim}_lstm_dropout={lstm_dropout}_learning_rate={learning_rate}.json"
    with open(output_file_name, "w") as f:
        json.dump(report, f, indent=4, default=convert)
    
