import pandas as pd
import logging

# Configuração do logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")
def create_inputs_txt(df, file_name):
    """
    Cria um arquivo inputs.txt com o texto dos laudos do DataFrame.

    Args:
        df: DataFrame do pandas contendo os dados dos laudos.
        file_name: Nome do arquivo para salvar os dados.
    """

    with open(file_name, "w", encoding="utf-8") as file:
        for _, row in df.iterrows():
            text= row["Texto"]
            id = int(float(row["Laudo"]))


            # Escreve no arquivo no formato desejado
            file.write(f"Dado o laudo: ({id}) {text}\n")
            file.write("Retornar a tabela do laudo preenchida no formato JSON:\n\n\n")

if __name__ == "__main__":
    # Ler um DataFrame do CSV
    df = pd.read_csv("llms/doc_similarity/data/test.csv")
    logging.info(f"DataFrame lido com {len(df)} laudos.")
    output_txt_file_name = "llms/zero_shot/data/inputs.txt"
    create_inputs_txt(df, output_txt_file_name)