import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon, kruskal
import matplotlib.pyplot as plt
import seaborn as sns
from uuid import uuid4

# Configuração para visualizações
sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})

# Função para calcular intervalo de confiança (95%)
def confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(n)
    margin = 1.96 * std_err  # Para 95% de confiança
    return mean, (mean - margin, mean + margin)

# Função para realizar análise estatística e gerar figuras
def analyze_f1_scores(data, task_name, model_names, results_dict):
    print(f"\n=== Análise para {task_name} ===")
    
    # Resumo estatístico
    summary = {}
    for model in model_names:
        mean, ci = confidence_interval(data[model])
        summary[model] = {'F1-Score Médio': mean, 'Intervalo de Confiança (95%)': ci}
    
    # Exibir resumo
    print("\nResumo Estatístico:")
    for model, stats in summary.items():
        print(f"{model}: F1-Score Médio = {stats['F1-Score Médio']:.3f}, IC 95% = ({stats['Intervalo de Confiança (95%)'][0]:.3f}, {stats['Intervalo de Confiança (95%)'][1]:.3f})")
    
    # Testes estatísticos
    p_values = {}
    if len(model_names) == 2:  # Para NER (BiLSTM-CRF vs BioBERTpt)
        # Teste t pareado
        stat_t, p_t = ttest_rel(data[model_names[0]], data[model_names[1]])
        print(f"\nTeste t pareado: p-valor = {p_t:.4f}")
        if p_t < 0.05:
            print(f"Resultado: Diferença significativa entre {model_names[0]} e {model_names[1]} (p < 0.05)")
        else:
            print(f"Resultado: Nenhuma diferença significativa entre {model_names[0]} e {model_names[1]} (p >= 0.05)")
        p_values['t-test'] = p_t
        
        # Teste de Wilcoxon (não paramétrico)
        stat_w, p_w = wilcoxon(data[model_names[0]], data[model_names[1]])
        print(f"Teste de Wilcoxon: p-valor = {p_w:.4f}")
        if p_w < 0.05:
            print(f"Resultado: Diferença significativa entre {model_names[0]} e {model_names[1]} (p < 0.05)")
        else:
            print(f"Resultado: Nenhuma diferença significativa entre {model_names[0]} e {model_names[1]} (p >= 0.05)")
        p_values['Wilcoxon'] = p_w
    
    else:  # Para QA (Gemini, GPT, LLaMA - zero-shot ou few-shot)
        # Teste de Kruskal-Wallis
        stat_k, p_k = kruskal(*[data[model] for model in model_names])
        print(f"\nTeste de Kruskal-Wallis: p-valor = {p_k:.4f}")
        if p_k < 0.05:
            print("Resultado: Pelo menos um modelo tem desempenho significativamente diferente (p < 0.05)")
        else:
            print("Resultado: Nenhuma diferença significativa entre os modelos (p >= 0.05)")
        p_values['Kruskal-Wallis'] = p_k
    
    # Armazenar resultados para tabela
    results_dict[task_name] = {'summary': summary, 'p_values': p_values}
    
    # Visualização: Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data[model_names])
    plt.title(f"Distribuição dos F1-Scores - {task_name}")
    plt.ylabel("F1-Score")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"f1_scores_boxplot_{task_name.lower().replace(' ', '_')}.png")
    plt.close()
    
    # Visualização: Gráfico de barras com intervalos de confiança
    means = [summary[model]['F1-Score Médio'] for model in model_names]
    cis = [summary[model]['Intervalo de Confiança (95%)'] for model in model_names]
    errors = [(mean - ci[0], ci[1] - mean) for mean, ci in zip(means, cis)]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, means, yerr=np.array(errors).T, capsize=5, color=sns.color_palette("Blues", len(model_names)))
    plt.title(f"F1-Scores Médios com Intervalos de Confiança - {task_name}")
    plt.ylabel("F1-Score Médio")
    plt.xticks(rotation=45, ha='right')
    for bar, mean in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width() / 2, mean + 0.01, f"{mean:.3f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(f"f1_scores_bar_{task_name.lower().replace(' ', '_')}.png")
    plt.close()

# Função para gerar tabela LaTeX com resultados
def generate_latex_table(results_dict):
    latex = """
\\documentclass{article}
\\usepackage{booktabs}
\\usepackage{amsmath}
\\begin{document}

\\begin{table}[h]
\\centering
\\caption{Resumo dos F1-Scores e Resultados Estatísticos}
\\begin{tabular}{lccc}
\\toprule
\\textbf{Tarefa} & \\textbf{Modelo} & \\textbf{F1-Score Médio (IC 95\\%)} & \\textbf{p-valor} \\\\
\\midrule
"""
    
    for task, result in results_dict.items():
        latex += f"\\multirow{{{len(result['summary'])}}}{{*}}{{{task}}} "
        for i, (model, stats) in enumerate(result['summary'].items()):
            f1_mean = stats['F1-Score Médio']
            ci = stats['Intervalo de Confiança (95%)']
            if i == 0:
                p_value_str = ", ".join([f"{k}: {v:.4f}" for k, v in result['p_values'].items()])
                latex += f"& {model} & {f1_mean:.3f} ({ci[0]:.3f}--{ci[1]:.3f}) & {p_value_str} \\\\ \n"
            else:
                latex += f"& {model} & {f1_mean:.3f} ({ci[0]:.3f}--{ci[1]:.3f}) & \\\\ \n"
        latex += "\\midrule\n"
    
    latex += """
\\bottomrule
\\end{tabular}
\\end{table}

\\end{document}
"""
    with open("f1_scores_summary_table.tex", "w") as f:
        f.write(latex)

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

# Criando o DataFrame com os F1-Scores
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

# Dicionário para armazenar resultados
results_dict = {}

# Realizar análise para NER
ner_models = ["BiLSTM-CRF", "BioBERTpt"]
analyze_f1_scores(f1_scores, "Reconhecimento de Entidade Nomeada (NER)", ner_models, results_dict)

# Realizar análise para QA Zero-Shot
qa_zero_shot_models = ["Gemini-1.5-Flash-Zero-Shot", "GPT-4o-Zero-Shot", "Llama 3 70B-Zero-Shot"]
analyze_f1_scores(f1_scores, "Resposta a Perguntas (QA Zero-Shot)", qa_zero_shot_models, results_dict)

# Realizar análise para QA Few-Shot
qa_few_shot_models = ["Gemini-1.5-Flash-Few-Shot", "GPT-4o-Few-Shot", "Llama 3 70B-Few-Shot"]
analyze_f1_scores(f1_scores, "Resposta a Perguntas (QA Few-Shot)", qa_few_shot_models, results_dict)

# Gerar tabela LaTeX
generate_latex_table(results_dict)

# Gerar relatório final
print("\n=== Relatório Final ===")

# Para NER
print("Para NER:")
if ttest_rel(f1_scores["BiLSTM-CRF"], f1_scores["BioBERTpt"])[1] < 0.05:
    better_model_ner = "BioBERTpt" if f1_scores["BioBERTpt"].mean() > f1_scores["BiLSTM-CRF"].mean() else "BiLSTM-CRF"
    print(f"O modelo {better_model_ner} é estatisticamente superior (p < 0.05).")
else:
    print("Nenhuma diferença significativa entre BiLSTM-CRF e BioBERTpt. Escolha com base em fatores práticos (e.g., custo, tempo).")

# Para QA Zero-Shot
print("\nPara QA Zero-Shot:")
if kruskal(f1_scores["Gemini-1.5-Flash-Zero-Shot"], f1_scores["GPT-4o-Zero-Shot"], f1_scores["Llama 3 70B-Zero-Shot"])[1] < 0.05:
    means = f1_scores[qa_zero_shot_models].mean()
    best_model_qa_zero = means.idxmax()
    print(f"O modelo {best_model_qa_zero} tem o maior F1-Score médio e é estatisticamente superior (p < 0.05).")
else:
    print("Nenhuma diferença significativa entre Gemini-1.5-Flash-Zero-Shot, GPT-4o-Zero-Shot e Llama 3 70B-Zero-Shot. Escolha com base em fatores práticos (e.g., custo, tempo).")

# Para QA Few-Shot
print("\nPara QA Few-Shot:")
if kruskal(f1_scores["Gemini-1.5-Flash-Few-Shot"], f1_scores["GPT-4o-Few-Shot"], f1_scores["Llama 3 70B-Few-Shot"])[1] < 0.05:
    means = f1_scores[qa_few_shot_models].mean()
    best_model_qa_few = means.idxmax()
    print(f"O modelo {best_model_qa_few} tem o maior F1-Score médio e é estatisticamente superior (p < 0.05).")
else:
    print("Nenhuma diferença significativa entre Gemini-1.5-Flash-Few-Shot, GPT-4o-Few-Shot e Llama 3 70B-Few-Shot. Escolha com base em fatores práticos (e.g., custo, tempo).")