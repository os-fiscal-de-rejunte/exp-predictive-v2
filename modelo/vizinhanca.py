import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from modelo.caracteristicas_grade import CLIMA

FONTES = {"mascara_foco_2d": "mascara", "viirs_registros_7d": "viirs", "modis_registros_7d": "modis"}
COLUNAS = ["viz_" + nome + sufixo for nome in FONTES.values() for sufixo in ["_soma", "_com_registro"]]


def adicionar(dados):
    dados = dados.sort_values(["data_utc", "linha", "coluna"]).copy().reset_index(drop=True)
    if dados.duplicated(["data_utc", "linha", "coluna"]).any():
        raise ValueError("Celula e data duplicadas ao calcular vizinhanca.")
    esperadas = np.tile(np.arange(256), dados["data_utc"].nunique())
    posicoes = dados["linha"].to_numpy() * 16 + dados["coluna"].to_numpy()
    if len(dados) != len(esperadas) or not np.array_equal(posicoes, esperadas):
        raise ValueError("A vizinhanca exige a grade completa de 256 celulas por dia.")
    for coluna, nome in FONTES.items():
        valores = pd.to_numeric(dados[coluna], errors="raise").to_numpy(dtype=float, na_value=np.nan).reshape(-1, 16, 16)
        ampliado = np.pad(valores, ((0, 0), (1, 1), (1, 1)), constant_values=np.nan)
        vizinhos = np.stack([ampliado[:, 1 + dy:17 + dy, 1 + dx:17 + dx]
                             for dy in [-1, 0, 1] for dx in [-1, 0, 1] if dy or dx])
        presentes = np.isfinite(vizinhos).sum(axis=0)
        soma = np.nansum(vizinhos, axis=0)
        soma[presentes == 0] = np.nan
        dados[f"viz_{nome}_soma"] = soma.reshape(-1)
        dados[f"viz_{nome}_com_registro"] = presentes.reshape(-1)
    dados["mes"] = pd.to_datetime(dados["data_utc"], utc=True).dt.month
    return dados


class AnomaliasClima(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        self.medias_ = {}
        for nome in CLIMA:
            coluna = "clima_" + nome
            valores = pd.to_numeric(x[coluna], errors="raise")
            self.medias_[coluna] = {"mensal": valores.groupby(x["mes"]).mean().to_dict(),
                                    "geral": float(valores.mean())}
        return self

    def transform(self, x):
        saida = x.copy()
        for coluna, medias in self.medias_.items():
            referencia = saida["mes"].map(medias["mensal"]).fillna(medias["geral"])
            saida[coluna + "_anomalia"] = saida[coluna] - referencia
        return saida.drop(columns="mes")
