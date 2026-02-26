import pandas as pd
from metrics import MetricsTracking

df_true_lung_rads = pd.read_csv("llms/doc_similarity/data/test_post_processed_final.csv", encoding="utf-8")
df_pred_lung_rads_few_shot_gemini = pd.read_csv("llms/few_shot/data/gemini_results/results_prompt_2_five_ex_structured_post_processed.csv")

# # Certifique-se de que estão ordenados pelo mesmo índice ou coluna "Laudo"
# df_true = df_true_lung_rads.sort_values(by="Laudo").reset_index(drop=True)
# df_pred = df_pred_lung_rads_few_shot_gemini.sort_values(by="Laudo").reset_index(drop=True)

# Remova a coluna "Laudo" e transforme cada linha em uma lista de rótulos
true_labels_list = df_true_lung_rads.drop(columns=["Laudo"]).values.tolist()
predicted_labels_list = df_pred_lung_rads_few_shot_gemini.drop(columns=["Id do laudo"]).values.tolist()
#predicted_labels_list = df_pred_lung_rads_few_shot_gemini.values.tolist()

# Use a classe para calcular métricas
tracker = MetricsTracking()
tracker.update(true_labels_list, predicted_labels_list)

# Obter as métricas por linha
metrics = tracker.get_all_metrics()

# Criar um novo DataFrame com as métricas
df_metrics = pd.DataFrame(metrics)

# Adicionar a coluna 'Laudo' correspondente
df_metrics.insert(0, "Laudo", df_true_lung_rads["Laudo"])

# Opcional: salvar como CSV
df_metrics.to_csv("eval_qa_models/gemini_few_shot_metrics.csv", index=False)