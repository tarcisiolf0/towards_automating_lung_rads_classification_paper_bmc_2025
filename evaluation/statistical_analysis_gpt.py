import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import friedmanchisquare, wilcoxon
from itertools import combinations
from statsmodels.stats.multitest import multipletests
import scikit_posthocs as sp

# Dados de entrada
bilstmcrf_metrics_df = pd.read_csv("evaluation/eval_ner_models/bilstm_metrics.csv", encoding="utf-8")
biobertpt_metrics_df = pd.read_csv("evaluation/eval_ner_models/biobertpt_metrics.csv", encoding="utf-8")
gemini_zero_shot_metrics_df = pd.read_csv("evaluation/eval_qa_models/gemini_zero_shot_metrics.csv", encoding="utf-8")
gpt_zero_shot_metrics_df = pd.read_csv("evaluation/eval_qa_models/gpt_zero_shot_metrics.csv", encoding="utf-8")
llama_zero_shot_metrics_df = pd.read_csv("evaluation/eval_qa_models/llama_zero_shot_metrics.csv", encoding="utf-8")
gemini_few_shot_metrics_df = pd.read_csv("evaluation/eval_qa_models/gemini_few_shot_metrics.csv", encoding="utf-8")
gpt_few_shot_metrics_df = pd.read_csv("evaluation/eval_qa_models/gpt_few_shot_metrics.csv", encoding="utf-8")
llama_few_shot_metrics_df = pd.read_csv("evaluation/eval_qa_models/llama_few_shot_metrics.csv", encoding="utf-8")

# Recuperando os F1-scores dos dataframes
bilstmcrf_f1 = bilstmcrf_metrics_df["f1"].values
biobertpt_f1 = biobertpt_metrics_df["f1"].values
gemini_zero_shot_f1 = gemini_zero_shot_metrics_df["f1"].values
gpt_zero_shot_f1 = gpt_zero_shot_metrics_df["f1"].values
llama_zero_shot_f1 = llama_zero_shot_metrics_df["f1"].values
gemini_few_shot_f1 = gemini_few_shot_metrics_df["f1"].values
gpt_few_shot_f1 = gpt_few_shot_metrics_df["f1"].values
llama_few_shot_f1 = llama_few_shot_metrics_df["f1"].values


f1_scores = pd.DataFrame({
    "BiLSTM-CRF": bilstmcrf_f1,
    "BioBERTpt": biobertpt_f1,
    "Gemini-1.5-Flash-Zero-Shot": gemini_zero_shot_f1,
    "GPT-4o-Zero-Shot": gpt_zero_shot_f1,
    "Llama 3 70B-Zero-Shot": llama_zero_shot_f1,
    "Gemini-1.5-Flash-Few-Shot": gemini_few_shot_f1,
    "GPT-4o-Few-Shot": gpt_few_shot_f1,
    "Llama 3 70B-Few-Shot": llama_few_shot_f1
})

# 1. Teste de Friedman
stat, p_friedman = friedmanchisquare(
    f1_scores["BiLSTM-CRF"],
    f1_scores["BioBERTpt"],
    f1_scores["Gemini-1.5-Flash-Zero-Shot"],
    f1_scores["GPT-4o-Zero-Shot"],
    f1_scores["Llama 3 70B-Zero-Shot"],
    f1_scores["Gemini-1.5-Flash-Few-Shot"],
    f1_scores["GPT-4o-Few-Shot"],
    f1_scores["Llama 3 70B-Few-Shot"]
)

print(f"Friedman test statistic: {stat:.4f}, p-value: {p_friedman:.4f}")

# 2. Se p < 0.05, aplica teste de Nemenyi
if p_friedman < 0.05:
    print("\nDiferença significativa encontrada. Executando teste de Nemenyi...\n")
    
    # Calcula os ranks por laudo
    ranks = f1_scores.rank(axis=1, method='average', ascending=False)
    
    # Aplica teste de Nemenyi
    nemenyi_result = sp.posthoc_nemenyi_friedman(ranks.values)
    nemenyi_result.columns = f1_scores.columns
    nemenyi_result.index = f1_scores.columns

    # Exibe matriz de p-valores
    print("\nMatriz de p-valores (teste de Nemenyi):\n")
    print(nemenyi_result.round(4))

    # 4. Visualização: Heatmap dos p-valores
    plt.figure(figsize=(8, 6))
    sns.heatmap(nemenyi_result, xticklabels=f1_scores.columns, yticklabels=f1_scores.columns,
                annot=True, cmap="coolwarm", fmt=".3f", cbar_kws={'label': 'p-valor'})
    plt.title("Teste de Nemenyi – Comparações entre Modelos")
    plt.tight_layout()
    plt.show()
else:
    print("Nenhuma diferença significativa encontrada entre os modelos (p ≥ 0.05).")

# 4. Visualização: Boxplot dos F1-scores
plt.figure(figsize=(10, 6))
sns.boxplot(data=f1_scores)
plt.title("Boxplot dos F1-Scores por Modelo")
plt.ylabel("F1-Score")
plt.xticks(rotation=45)
plt.grid(True)

# 5. Calcular e exibir mediana e desvio padrão para cada modelo
for i, col in enumerate(f1_scores.columns):
    median = f1_scores[col].median()
    std = f1_scores[col].std()
    
    # Adicionar texto para a mediana e desvio padrão
    plt.text(i, median + 0.02, f'Median: {median:.2f}', ha='center', color='black', fontsize=10)
    plt.text(i, median - 0.02, f'SD: {std:.2f}', ha='center', color='black', fontsize=10)

plt.tight_layout()
plt.show()