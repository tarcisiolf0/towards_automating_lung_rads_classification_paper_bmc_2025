import pandas as pd
from seqeval.metrics import classification_report, f1_score
import numpy as np
from metrics import MetricsTracking

def read_csv(file_path):
    """
    Função para ler um arquivo CSV e retornar um DataFrame.
    """
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
        return df
    except Exception as e:
        print(f"Erro ao ler o arquivo {file_path}: {e}")
        return None

def calculate_metrics(bilstm_predictions, biobertpt_predictions):
    tracker_bilstm = MetricsTracking()
    tracker_biobertpt = MetricsTracking()

    # Rótulos verdadeiros (iguais para ambos)
    true_labels_bilstm = group_by_idx(bilstm_predictions, "iob_tag")
    true_labels_biobertpt = group_by_idx(biobertpt_predictions, "iob_tag")

    # Rótulos preditos
    pred_labels_bilstm = group_by_idx(bilstm_predictions, "predicted_iob_tag")
    pred_labels_biobertpt = group_by_idx(biobertpt_predictions, "predicted_iob_tag")

    # Atualizar as métricas
    tracker_bilstm.update(true_labels_bilstm, pred_labels_bilstm)
    tracker_biobertpt.update(true_labels_biobertpt, pred_labels_biobertpt)

    # Obter as métricas individuais
    metrics_bilstm = tracker_bilstm.get_all_metrics()
    metrics_biobertpt = tracker_biobertpt.get_all_metrics()

    # Converter para DataFrame se quiser salvar ou analisar
    df_bilstm = pd.DataFrame(metrics_bilstm)
    df_biobertpt = pd.DataFrame(metrics_biobertpt)

    # Exemplo: mostrar média dos F1-scores
    print("F1 BILSTM (média):", df_bilstm["f1"].mean())
    print("F1 BioBERTpt (média):", df_biobertpt["f1"].mean())

    df_bilstm.to_csv("eval_ner_models/bilstm_metrics.csv", index=False)
    df_biobertpt.to_csv("eval_ner_models/biobertpt_metrics.csv", index=False)


def calculate_metrics_seqeval_each_report(bilstm_predictions, biobertpt_predictions):
    # Calcular F1-score para cada relatório

    for idx in unique_index:
        # Filtrar os DataFrames para o índice atual
        bilstm_filtered = bilstm_predictions[bilstm_predictions["report_idx"] == idx]
        biobertpt_filtered = biobertpt_predictions[biobertpt_predictions["report_idx"] == idx]

        # Agrupar por rótulo
        true_labels_bilstm = group_by_idx(bilstm_filtered, "iob_tag")
        pred_labels_bilstm = group_by_idx(bilstm_filtered, "predicted_iob_tag")

        true_labels_biobertpt = group_by_idx(biobertpt_filtered, "iob_tag")
        pred_labels_biobertpt = group_by_idx(biobertpt_filtered, "predicted_iob_tag")

        # Calcular F1-score
        # print(true_labels_bilstm)
        # print("\n\n")
        # print(pred_labels_bilstm)

        bilstm_classification_report = classification_report(true_labels_bilstm, pred_labels_bilstm, output_dict=True)
        biobertpt_classification_report = classification_report(true_labels_biobertpt, pred_labels_biobertpt, output_dict=True)

        bilstm_f1 = bilstm_classification_report["macro avg"]["f1-score"]
        biobertpt_f1 = biobertpt_classification_report["macro avg"]["f1-score"]

        f1_scores_bilstm.append(bilstm_f1)
        f1_scores_biobertpt.append(biobertpt_f1)

    # Resumir os resultados

    # Remove NaN values
    f1_scores_bilstm = [score for score in f1_scores_bilstm if not pd.isna(score)]
    f1_scores_biobertpt = [score for score in f1_scores_biobertpt if not pd.isna(score)]
    # Calcular a média dos F1-scores
    print("F1 BILSTM (média):", round(np.mean(f1_scores_bilstm), 2))
    print("F1 BioBERTpt (média):", round(np.mean(f1_scores_biobertpt), 2))

    print("Len BILSTM:", len(f1_scores_bilstm))
    print("Len BioBERTpt:", len(f1_scores_biobertpt))


def calculate_metrics_seqeval(bilstm_predictions, biobertpt_predictions, unique_index):
    bilstm_filtered = bilstm_predictions[bilstm_predictions["report_idx"].isin(unique_index)]
    biobertpt_filtered = biobertpt_predictions[biobertpt_predictions["report_idx"].isin(unique_index)]

    # Agrupar por rótulo
    true_labels_bilstm = bilstm_filtered["iob_tag"].tolist()
    pred_labels_bilstm = bilstm_filtered["predicted_iob_tag"].tolist()
    true_labels_biobertpt = biobertpt_filtered["iob_tag"].tolist()
    pred_labels_biobertpt = biobertpt_filtered["predicted_iob_tag"].tolist()
    # Calcular Classification Report
    bilstm_classification_report = classification_report([true_labels_bilstm], [pred_labels_bilstm], output_dict=True)
    biobertpt_classification_report = classification_report([true_labels_biobertpt], [pred_labels_biobertpt], output_dict=True)
    # Imprimir os resultados
    print("BILSTM Classification Report:")
    print("Precision: ", round(bilstm_classification_report["macro avg"]["precision"],2))
    print("Recall: ", round(bilstm_classification_report["macro avg"]["recall"],2))
    print("F1: ", round(bilstm_classification_report["macro avg"]["f1-score"],2))

    print("BioBERTpt Classification Report:")
    print("Precision: ", round(biobertpt_classification_report["macro avg"]["precision"],2))
    print("Recall: ", round(biobertpt_classification_report["macro avg"]["recall"],2))
    print("F1: ", round(biobertpt_classification_report["macro avg"]["f1-score"],2))

# Função auxiliar para agrupar rótulos por relatório
def group_by_idx(df, label_col):
    return [group[label_col].tolist() for _, group in df.groupby("report_idx")]


if __name__ == "__main__":
    # Caminhos para os arquivos CSV
    bilstmcrf_filename = "bilstmcrf_pytorch/data/lung_rads/lung_rads_test_predicted.csv"
    biobertpt_filename = "biobertpt/data/lung_rads/lung_rads_test_predicted.csv"

    bilstm_predictions = read_csv(bilstmcrf_filename)
    biobertpt_predictions = read_csv(biobertpt_filename)

    f1_scores_bilstm = []
    f1_scores_biobertpt = []

    # Obter índice únicos dos relatórios
    unique_index = bilstm_predictions["report_idx"].unique().tolist()

    # Cacular métricas para cada relatório
    #calculate_metrics(bilstm_predictions, biobertpt_predictions)
    calculate_metrics_seqeval(bilstm_predictions, biobertpt_predictions, unique_index)
    