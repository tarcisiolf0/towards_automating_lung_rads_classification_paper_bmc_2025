import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import friedmanchisquare, wilcoxon
from itertools import combinations
from statsmodels.stats.multitest import multipletests
import scikit_posthocs as sp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp


def print_metrics(df, model_name):
    """
    Função para imprimir as métricas de precisão, recall e F1-score.
    """
    print("\n" + "="*50)
    print(f"Métricas de Avaliação: {model_name}")
    print("="*50)
    print(f"Precisão: {df['precision'].mean():.2f}")
    print(f"Recall: {df['recall'].mean():.2f}")
    print(f"F1-Score: {df['f1'].mean():.2f}")
    
# Configuração de estilo para os gráficos
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# Dados de entrada
#bilstmcrf_metrics_df = pd.read_csv("evaluation/eval_ner_models/bilstm_metrics.csv", encoding="utf-8")
#biobertpt_metrics_df = pd.read_csv("evaluation/eval_ner_models/biobertpt_metrics.csv", encoding="utf-8")
gemini_zero_shot_metrics_df = pd.read_csv("evaluation/eval_qa_models/gemini_zero_shot_metrics.csv", encoding="utf-8")
gpt_zero_shot_metrics_df = pd.read_csv("evaluation/eval_qa_models/gpt_zero_shot_metrics.csv", encoding="utf-8")
llama_zero_shot_metrics_df = pd.read_csv("evaluation/eval_qa_models/llama_zero_shot_metrics.csv", encoding="utf-8")
gemini_few_shot_metrics_df = pd.read_csv("evaluation/eval_qa_models/gemini_few_shot_metrics.csv", encoding="utf-8")
gpt_few_shot_metrics_df = pd.read_csv("evaluation/eval_qa_models/gpt_few_shot_metrics.csv", encoding="utf-8")
llama_few_shot_metrics_df = pd.read_csv("evaluation/eval_qa_models/llama_few_shot_metrics.csv", encoding="utf-8")

# Recuperando os F1-scores dos dataframes
#bilstmcrf_f1 = bilstmcrf_metrics_df["f1"].values
#biobertpt_f1 = biobertpt_metrics_df["f1"].values
gemini_zero_shot_f1 = gemini_zero_shot_metrics_df["f1"].values
gpt_zero_shot_f1 = gpt_zero_shot_metrics_df["f1"].values
llama_zero_shot_f1 = llama_zero_shot_metrics_df["f1"].values
gemini_few_shot_f1 = gemini_few_shot_metrics_df["f1"].values
gpt_few_shot_f1 = gpt_few_shot_metrics_df["f1"].values
llama_few_shot_f1 = llama_few_shot_metrics_df["f1"].values


# data = pd.DataFrame({
#     "BiLSTM-CRF": bilstmcrf_f1,
#     "BioBERTpt": biobertpt_f1,
#     "Gemini-1.5-Flash-Zero-Shot": gemini_zero_shot_f1,
#     "GPT-4o-Zero-Shot": gpt_zero_shot_f1,
#     "Llama 3 70B-Zero-Shot": llama_zero_shot_f1,
#     "Gemini-1.5-Flash-Few-Shot": gemini_few_shot_f1,
#     "GPT-4o-Few-Shot": gpt_few_shot_f1,
#     "Llama 3 70B-Few-Shot": llama_few_shot_f1
# })

data = pd.DataFrame({
    "Gemini-1.5-Flash-Zero-Shot": gemini_zero_shot_f1,
    "GPT-4o-Zero-Shot": gpt_zero_shot_f1,
    "Llama 3 70B-Zero-Shot": llama_zero_shot_f1,
    "Gemini-1.5-Flash-Few-Shot": gemini_few_shot_f1,
    "GPT-4o-Few-Shot": gpt_few_shot_f1,
    "Llama 3 70B-Few-Shot": llama_few_shot_f1
})

df = pd.DataFrame(data)

# =============================================
# 1. Análise Descritiva
# =============================================

#print_metrics(bilstmcrf_metrics_df, "BiLSTM-CRF")
#print("="*50)
#print_metrics(biobertpt_metrics_df, "BioBERTpt")
#print("="*50)
print_metrics(gemini_zero_shot_metrics_df, "Gemini-1.5-Flash-Zero-Shot")
print("="*50)
print_metrics(gemini_few_shot_metrics_df, "Gemini-1.5-Flash-Few-Shot")
print("="*50)
print_metrics(gpt_zero_shot_metrics_df, "GPT-4o-Zero-Shot")
print("="*50)
print_metrics(gpt_few_shot_metrics_df, "GPT-4o-Few-Shot")
print("="*50)
print_metrics(llama_zero_shot_metrics_df, "Llama 3 70B-Zero-Shot")
print("="*50)
print_metrics(llama_few_shot_metrics_df, "Llama 3 70B-Few-Shot")

# =============================================
# 2. Análise Descritiva
# =============================================
print("="*50)
print("Estatísticas Descritivas dos F1-Scores por Modelo")
print("="*50)
print(df.describe().round(3))

# Gráfico de distribuição
plt.figure(figsize=(12, 6))
sns.boxplot(data=df)
plt.title("Distribution of F1-Scores by LLM", fontsize=16)
plt.ylabel("F1-Score", fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

# =============================================
# 3. Teste de Normalidade (Shapiro-Wilk)
# =============================================
print("\n" + "="*50)
print("Teste de Normalidade (Shapiro-Wilk)")
print("="*50)
for model in df.columns:
    stat, p = stats.shapiro(df[model])
    print(f"{model:10} -> p-valor = {p:.4f} {'(Normal)' if p > 0.05 else '(Não normal)'}")

# =============================================
# 4. Teste de Friedman (não paramétrico para múltiplos grupos dependentes)
# =============================================
print("\n" + "="*50)
print("Teste de Friedman (Comparação Global)")
print("="*50)
stat, p = stats.friedmanchisquare(*[df[model] for model in df.columns])
print(f"Estatística Friedman: {stat:.3f}, p-valor = {p:.4f}")
if p < 0.05:
    print("Há diferenças significativas entre os modelos (p < 0.05)")
else:
    print("Não há diferenças significativas entre os modelos")

# =============================================
# 5. Post-hoc Analysis (Teste de Nemenyi)
# =============================================
if p < 0.05:  # Só executa se Friedman for significativo
    print("\n" + "="*50)
    print("Teste Post-hoc de Nemenyi (Dados Pareados)")
    print("="*50)
    
    # # Prepara os dados no formato longo COM IDENTIFICAÇÃO DA AMOSTRA
    # melted_df = df.reset_index().melt(id_vars='index', var_name='Modelo', value_name='F1-Score')
    # melted_df.rename(columns={'index': 'Amostra'}, inplace=True)
    
    # Executa o Nemenyi para dados pareados
    # nemenyi = sp.posthoc_nemenyi_friedman(melted_df, y_col='F1-Score', group_col='Modelo', block_col='Amostra')
    
    # Calcula os ranks por laudo
    ranks = data.rank(axis=1, method='average', ascending=False)
    
    # Aplica teste de Nemenyi
    nemenyi_result = sp.posthoc_nemenyi_friedman(ranks.values)
    nemenyi_result.columns = data.columns
    nemenyi_result.index = data.columns
    print("Matriz de p-valores (Nemenyi):")
    print(nemenyi_result.round(4))
    
    # # Visualização térmica
    # plt.figure(figsize=(10, 8))
    # sp.sign_plot(nemenyi_result, **{'linewidths': 0.25, 'linecolor': '0.5', 'square': True})
    # plt.title("Comparações Par a Par (Nemenyi Pareado)", pad=20)
    # plt.tight_layout()
    # plt.show()
    # 4. Visualização: Heatmap dos p-valores
    plt.figure(figsize=(8, 6))
    sns.heatmap(nemenyi_result, xticklabels=data.columns, yticklabels=data.columns,
                annot=True, cmap="coolwarm", fmt=".3f", cbar_kws={'label': 'p-value'})
    plt.title("Nemenyi Test – Comparison between LLMs", fontsize=16)
    plt.tight_layout()
    plt.show()

# =============================================
# 6. Comparação Direta entre os 2 Melhores Modelos (Wilcoxon pareado)
# =============================================
# Identifica os 2 modelos com maiores medianas
top_models = df.median().sort_values(ascending=False).index[:2]
print("\n" + "="*50)
print(f"Comparação Pareada entre os Melhores Modelos: {top_models[0]} vs {top_models[1]}")
print("="*50)

stat, p = stats.wilcoxon(df[top_models[0]], df[top_models[1]])
print(f"Teste de Wilcoxon: p-valor = {p:.4f}")
if p < 0.05:
    print(f"Diferença significativa: {top_models[0]} é melhor que {top_models[1]}")
else:
    print(f"Não há diferença significativa entre {top_models[0]} e {top_models[1]}")

# =============================================
# 7. Gráfico Final de Desempenho Comparado
# =============================================
plt.figure(figsize=(12, 6))
sns.pointplot(data=df.melt(var_name='Modelo', value_name='F1-Score'), 
              x='Modelo', y='F1-Score', 
              estimator=np.median, errorbar=('ci', 95), 
              capsize=0.1, linestyle='none')
plt.title("Comparação dos Modelos (Mediana e IC 95%)", fontsize=14)
plt.ylim(0.6, 1.0)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()