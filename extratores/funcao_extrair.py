import csv
import os
from pathlib import Path

from dotenv import load_dotenv
import psycopg

RAIZ = Path(__file__).resolve().parent.parent
load_dotenv(RAIZ / ".env")
banco = os.getenv("BANCO")


def conectar_banco():
    if not banco:
        raise ValueError("A variável BANCO não foi definida no arquivo .env.")
    return psycopg.connect(banco)


def salvar_csv_banco(dados, nome_arquivo, colunas):
    caminho_arquivo = RAIZ / "dados" / nome_arquivo
    caminho_arquivo.parent.mkdir(parents=True, exist_ok=True)
    with caminho_arquivo.open("w", encoding="utf-8", newline="") as arquivo_csv:
        escritor = csv.writer(arquivo_csv)
        escritor.writerow(colunas)
        escritor.writerows(dados)
    print(f"Arquivo CSV '{nome_arquivo}' salvo em '{caminho_arquivo}'.")


def extrair_dados_harmonizados_banco():
    with conectar_banco() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM curated_ml_feature_store;")
            colunas = [coluna.name for coluna in cur.description]
            dados_satelites = cur.fetchall()
    salvar_csv_banco(dados_satelites, "dados_satelites.csv", colunas)
    return dados_satelites


def extrair_dados_previsao():
    with conectar_banco() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM curated_harmonized_events;")
            colunas = [coluna.name for coluna in cur.description]
            dados_previsao = cur.fetchall()
    salvar_csv_banco(dados_previsao, "dados_previsao.csv", colunas)
    return dados_previsao
