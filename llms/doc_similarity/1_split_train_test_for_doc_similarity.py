import pandas as pd
import json

def read_odf_file(filename):
    df = pd.read_excel(filename, engine="odf")
    return df

def list_of_dicts_csv_file(df, list_samples):
    for idx, row in df.iterrows():
        list_samples.append(
            {
            "id": int(row["Laudo"]),
            "text": row["Texto"]
            }
                        )
    return list_samples

def indexes_train_test(df_bert_train, df_bert_test):

    indexes_train = df_bert_train['report'].unique()
    indexes_test = df_bert_test['report'].unique()

    return indexes_train, indexes_test

def train_test_csv_files(filename, indexes_train, indexes_test):
    df = read_odf_file(filename)

    df_train = df.loc[df["Laudo"].isin(indexes_train)]
    df_test = df.loc[df["Laudo"].isin(indexes_test)]
    
    df_train.to_csv("llms/doc_similarity/data/train.csv", index=False)
    df_test.to_csv("llms/doc_similarity/data/test.csv", index=False)

def train_test_json_file(train_output_filename, test_output_filename):
    df_train = pd.read_csv("llms/doc_similarity/data/train.csv", usecols = ['Laudo', 'Texto'])
    df_test = pd.read_csv("llms/doc_similarity/data/test.csv", usecols = ['Laudo', 'Texto'])

    train_samples = []
    test_samples = []

    train_samples = list_of_dicts_csv_file(df_train, train_samples)
    test_samples = list_of_dicts_csv_file(df_test, test_samples)

    json.dump(train_samples, open(train_output_filename, encoding='utf-8', mode='w'), ensure_ascii=False, sort_keys=True, indent = 2)
    json.dump(test_samples, open(test_output_filename, encoding='utf-8', mode='w'), ensure_ascii=False, sort_keys=True, indent = 2)

def csv_to_json(csv_filename, output_filename):
    
    list_samples = []
    df = pd.read_csv(csv_filename, usecols=['Laudo',
                                     'Texto', 
                                     'O nódulo é sólido ou em partes moles?', 
                                     'O nódulo tem densidade semissólida ou parcialmente sólida?', 
                                     'O nódulo tem densidade em vidro fosco?',
                                     'O nódulo tem borda espiculada e/ou mal definida?', 
                                     'O nódulo é calcificado?', 
                                     'Localização do nódulo',
                                     'Tamanho do nódulo'])
    for idx, row in df.iterrows():
        list_samples.append(
        {
        "Id do laudo" : int(row["Laudo"]),
        "O nódulo é sólido ou em partes moles?" : row["O nódulo é sólido ou em partes moles?"],
        "O nódulo tem densidade semissólida ou parcialmente sólida?" : row["O nódulo tem densidade semissólida ou parcialmente sólida?"],
        "O nódulo é em vidro fosco?" : row["O nódulo tem densidade em vidro fosco?"],
        "O nódulo é espiculado, irregular ou mal definido?" : row["O nódulo tem borda espiculada e/ou mal definida?"],
        "O nódulo é calcificado?" : row["O nódulo é calcificado?"],
        "Localização do nódulo" : row["Localização do nódulo"],
        "Tamanho do nódulo" : row["Tamanho do nódulo"]
        }
                    )

    json.dump(list_samples, open(output_filename, encoding='utf-8', mode='w'), ensure_ascii=False, sort_keys=False, indent = 2)
    return


if __name__ == '__main__':
    # Step 1 - Split the data into train and test csv files
    odf_filename = "llms/doc_similarity/data/structured_data_for_lung_rads.ods"
    train_df_filename = "biobertpt/data/df_train_tokens_labeled_iob_bert_format.csv"
    test_df_filename = "biobertpt/data/df_test_tokens_labeled_iob_bert_format.csv"


    train_df = pd.read_csv(train_df_filename, encoding='utf-8')
    test_df = pd.read_csv(test_df_filename, encoding='utf-8')

    indexes_train, indexes_test = indexes_train_test(train_df, test_df)
    train_test_csv_files(odf_filename, indexes_train, indexes_test)

    # # Step 2 - Create json files with train and test reports
    train_output_filename = "llms/doc_similarity/data/train_samples.json"
    test_output_filename = "llms/doc_similarity/data/test_samples.json"
    train_test_json_file(train_output_filename, test_output_filename)

    # # Step 3 - Create json file with the question table from the training data
    train_csv_filename = "llms/doc_similarity/data/train.csv"
    train_output_examples_filename = "llms/doc_similarity/data//train_tabels.json"
    csv_to_json(train_csv_filename, train_output_examples_filename)

