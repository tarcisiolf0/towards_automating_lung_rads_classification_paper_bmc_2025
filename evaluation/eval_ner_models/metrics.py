from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class MetricsTracking:
    """
    Classe para calcular métricas de classificação individualmente para cada par de listas
    (rótulos verdadeiros e preditos) e armazenar os resultados.
    """

    def __init__(self):
        self.metrics_list = []

    def update(self, true_labels_list, predicted_labels_list):
        """
        Para cada par de listas internas, calcula as métricas e armazena os resultados.
        """
        assert len(true_labels_list) == len(predicted_labels_list), "Listas de entrada devem ter o mesmo tamanho."

        for true_labels, pred_labels in zip(true_labels_list, predicted_labels_list):
            assert len(true_labels) == len(pred_labels), "As listas de rótulos devem ter o mesmo comprimento."

            acc = accuracy_score(true_labels, pred_labels)
            f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0.0)
            precision = precision_score(true_labels, pred_labels, average="macro", zero_division=0.0)
            recall = recall_score(true_labels, pred_labels, average="macro", zero_division=0.0)

            self.metrics_list.append({
                "acc": round(acc, 3),
                "f1": round(f1, 3),
                "precision": round(precision, 3),
                "recall": round(recall, 3)
            })

    def get_all_metrics(self):
        """
        Retorna a lista com as métricas de cada amostra.
        """
        return self.metrics_list