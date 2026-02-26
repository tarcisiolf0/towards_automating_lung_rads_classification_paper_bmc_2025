from classification import classify_nodules
import pandas as pd
from sklearn.metrics import classification_report

if __name__ == '__main__':

    # Carregar os DataFrames
    df_results_bilstmcrf = pd.read_csv("bilstmcrf_pytorch/data/lung_rads/lung_rads_test_predicted_post_processed.csv", encoding="utf-8")
    df_results_biobertpt = pd.read_csv("biobertpt/data/lung_rads/lung_rads_test_predicted_post_processed.csv", encoding="utf-8")

    df_results_gemini = pd.read_csv("llms/few_shot/data/gemini_results/results_prompt_2_five_ex_structured_post_processed.csv", encoding="utf-8")
    df_results_gpt = pd.read_csv("llms/few_shot/data/gpt_results/results_prompt_2_five_ex_structured_post_processed.csv", encoding="utf-8")
    df_results_llama = pd.read_csv("llms/few_shot/data/llama_results/results_prompt_2_five_ex_structured_post_processed.csv", encoding="utf-8")
    


    test_df = pd.read_csv("llms/doc_similarity/data/test_final.csv", encoding="utf-8")
    list_lung_rads = test_df['Lung-RADS'].to_list()
    
    for i in range(len(list_lung_rads)):
        item = list_lung_rads[i]
        if item == 0.0: list_lung_rads[i] = "0"
        elif item == 1.0: list_lung_rads[i] = "1"
        elif item == 2.0: list_lung_rads[i] = "2"
        elif item == 3.0: list_lung_rads[i] = "3"
        elif item == 4.0: list_lung_rads[i] = "4A"
        elif item == 5.0: list_lung_rads[i] = "4B"
        elif item == 6.0: list_lung_rads[i] = "4X"
    

    df_results_bilstmcrf.insert(8 ,"Lung-RADS", list_lung_rads)
    df_results_biobertpt.insert(8 ,"Lung-RADS", list_lung_rads)
    df_results_gemini.insert(8 ,"Lung-RADS", list_lung_rads)
    df_results_gpt.insert(8 ,"Lung-RADS", list_lung_rads)
    df_results_llama.insert(8 ,"Lung-RADS", list_lung_rads)

    # # Classificar os nódulos nos DataFrames
    df_results_bilstmcrf['Lung-RADS-Pred'] = classify_nodules(df_results_bilstmcrf)
    df_results_biobertpt['Lung-RADS-Pred'] = classify_nodules(df_results_biobertpt)
    df_results_gemini['Lung-RADS-Pred'] = classify_nodules(df_results_gemini)
    df_results_gpt['Lung-RADS-Pred'] = classify_nodules(df_results_gpt)
    df_results_llama['Lung-RADS-Pred'] = classify_nodules(df_results_llama)

    # Exibir o DataFrame com as classificações
    # for row in df_results_bilstmcrf.iterrows():
    #     print(row[1]["Nódulo"], row[1]['Lung-RADS'], row[1]['Lung-RADS-Pred'], row[1]["Tamanho do nódulo (mm)"])
        
    print(df_results_biobertpt[['Lung-RADS', 'Lung-RADS-Pred']])
    print(df_results_gemini[['Lung-RADS', 'Lung-RADS-Pred']])
    print(df_results_gpt[['Lung-RADS', 'Lung-RADS-Pred']])
    print(df_results_llama[['Lung-RADS', 'Lung-RADS-Pred']])    

    y_true = list_lung_rads
    y_pred_bilstmcrf = df_results_bilstmcrf['Lung-RADS-Pred'].tolist()
    y_pred_biobertpt = df_results_biobertpt['Lung-RADS-Pred'].tolist()
    y_pred_gemini = df_results_gemini['Lung-RADS-Pred'].tolist()
    y_pred_gpt = df_results_gpt['Lung-RADS-Pred'].tolist()
    y_pred_llama = df_results_llama['Lung-RADS-Pred'].tolist()

    y_true_str = [str(x) for x in y_true]

    y_pred_bilstmcrf = [str(item) for item in y_pred_bilstmcrf]
    y_pred_biobertpt = [str(item) for item in y_pred_biobertpt]
    y_pred_gemini = [str(item) for item in y_pred_gemini]
    y_pred_gpt = [str(item) for item in y_pred_gpt]
    y_pred_llama = [str(item) for item in y_pred_llama]


    metrics_bilstmcrf = classification_report(y_true_str, y_pred_bilstmcrf, zero_division=0.0, output_dict=True)
    metrics_biobertpt = classification_report(y_true_str, y_pred_biobertpt, zero_division=0.0, output_dict=True)
    metrics_gemini = classification_report(y_true_str, y_pred_gemini, zero_division=0.0, output_dict=True)
    metrics_gpt = classification_report(y_true_str, y_pred_gpt, zero_division=0.0, output_dict=True)
    metrics_llama = classification_report(y_true_str, y_pred_llama, zero_division=0.0, output_dict=True)

    df_metrics_bilstmcrf = pd.DataFrame(data=metrics_bilstmcrf).transpose()
    df_metrics_bioberpt = pd.DataFrame(data=metrics_biobertpt).transpose()
    df_metrics_gemini = pd.DataFrame(data=metrics_gemini).transpose()
    df_metrics_gpt = pd.DataFrame(data=metrics_gpt).transpose()
    df_metrics_llama = pd.DataFrame(data=metrics_llama).transpose()

    df_metrics_bilstmcrf.to_csv('lung_rads_calc/data/lung_rads_metrics_bilstmcrf.csv')
    df_metrics_bioberpt.to_csv('lung_rads_calc/data/lung_rads_metrics_bioberpt.csv')
    df_metrics_gemini.to_csv('lung_rads_calc/data/lung_rads_metrics_gemini.csv')
    df_metrics_gpt.to_csv('lung_rads_calc/data/lung_rads_metrics_gpt.csv')
    df_metrics_llama.to_csv('lung_rads_calc/data/lung_rads_metrics_llama.csv')

    df_results_bilstmcrf.to_csv('lung_rads_calc/data/lung_rads_predictions_bilstmcrf.csv', index=False)
    df_results_biobertpt.to_csv('lung_rads_calc/data/lung_rads_predictions_bioberpt.csv', index=False)
    df_results_gemini.to_csv('lung_rads_calc/data/lung_rads_predictions_gemini.csv', index=False)
    df_results_gpt.to_csv('lung_rads_calc/data/lung_rads_predictions_gpt.csv', index=False)
    df_results_llama.to_csv('lung_rads_calc/data/lung_rads_predictions_llama.csv', index=False)
