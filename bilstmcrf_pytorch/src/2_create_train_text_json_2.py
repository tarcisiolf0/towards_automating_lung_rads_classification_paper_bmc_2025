import sys
import os

# Obtém o caminho absoluto para o diretório pai de 'src'
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Adiciona o diretório raiz ao sys.path se ele ainda não estiver lá
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)



import pandas as pd
from sklearn.model_selection import train_test_split
from utils.data_processing import create_mappings, save_json, save_txt
import random


def preprocess_ner_data(incomplete_csv_file_path, lr_csv_file_path, test_size, random_state):
    """ 
    Preprocesses the NER data from CSV files and splits it into training and testing sets.
    Args:
        incomplete_csv_file_path (str): Path to the CSV file containing incomplete data.
        lr_csv_file_path (str): Path to the CSV file containing LungRADS data.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        list of tuples: A list of tuple containing the training sentences, testing sentences, and unique tags.
    """

    # Read the CSV files
    df_incomplete = pd.read_csv(incomplete_csv_file_path, encoding='utf-8')
    grouped = df_incomplete.groupby('report_index')
    sentences = []

    # Iterate through the groups and create sentences
    for report_index, group in grouped:
        sentence = list(zip(group['token'], group['iob_tag']))
        sentences.append({"report_index": report_index, "sentence": sentence})  
    
    # Read the LungRADS CSV file
    df_lr = pd.read_csv(lr_csv_file_path, encoding='utf-8')
    grouped_lr = df_lr.groupby('report_index')
    sentences_lr = []

    # Iterate through the groups and create sentences
    for report_index, group in grouped_lr:
        sentence = list(zip(group['token'], group['iob_tag']))
        sentences_lr.append({"report_index": report_index, "sentence": sentence})  

    unique_tags = sorted(list(set([tag for item in sentences for token, tag in item['sentence']])))

    if '<PAD>' not in unique_tags:
        unique_tags.append('<PAD>')
    if '<UNK>' not in unique_tags:
        unique_tags.append('<UNK>')


    train_sentences, test_sentences_temp = train_test_split(sentences, test_size=test_size, random_state=random_state)

    test_sentences = []
    for tuple_incomplete in test_sentences_temp:
        test_sentences.append(tuple_incomplete)

    for tuple_lr in sentences_lr:
        test_sentences.append(tuple_lr)

    return train_sentences, test_sentences, unique_tags

if __name__ == '__main__':
 
    folder_path = "bilstmcrf_pytorch/data/"
    incomplete_csv_file_path = "bilstmcrf_pytorch/data/df_tokens_labeled_iob_without_lr_idx.csv"
    lr_csv_file_path = "bilstmcrf_pytorch/data/df_tokens_labeled_iob_with_lr_idx.csv"

    num = random.randrange(42000)
    save_txt(num, folder_path+"random_number.txt")

    train_data, test_data, unique_tags = preprocess_ner_data(incomplete_csv_file_path,
                                                             lr_csv_file_path,
                                                             test_size=0.219,
                                                             random_state=num)
    
    word2index, index2word, tag2index, index2tag = create_mappings([item['sentence'] for item in train_data + test_data], unique_tags) # Pass sentences only

    save_json(train_data, folder_path+"train_data.json")
    save_json(test_data, folder_path+"test_data.json")
    save_json(unique_tags, folder_path+"unique_tags.json")
    save_json(word2index, folder_path+"word2index.json")
    save_json({str(k): v for k, v in index2word.items()}, folder_path+"index2word.json")
    save_json(tag2index, folder_path+"tag2index.json")
    save_json({str(k): v for k, v in index2tag.items()}, folder_path+"index2tag.json")
    
