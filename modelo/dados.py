from pathlib import Path

import numpy as np
import pandas as pd

from modelo.config import ALVO, ENTRADAS, NUMERICAS

COLUNAS = [
    "grid_id", "acq_date", "has_valid_satellite_obs", "country",
    *[c for c in NUMERICAS if c not in {"dia_seno", "dia_cosseno"}],
]


def ler_dados(caminhos):
    partes = []
    for caminho in caminhos:
        arquivo = Path(caminho)
        dados = pd.read_csv(arquivo, dtype={"grid_id": "string", "country": "string"})
        ausentes = sorted(set(COLUNAS) - set(dados.columns))
        if ausentes:
            raise ValueError(f"Colunas ausentes em {arquivo}: {ausentes}")
        partes.append(dados[COLUNAS].copy())
    if not partes:
        raise ValueError("Informe ao menos um arquivo de dados.")
    dados = pd.concat(partes, ignore_index=True)
    dados["acq_date"] = pd.to_datetime(dados["acq_date"], format="mixed", utc=True, errors="raise").dt.normalize()
    if dados["acq_date"].isna().any() or dados["grid_id"].isna().any():
        raise ValueError("Toda linha precisa de grid_id e acq_date.")
    if dados.duplicated(["grid_id", "acq_date"]).any():
        raise ValueError("Existem registros duplicados para a mesma celula e data.")
    valores = dados["has_valid_satellite_obs"].astype("string").str.strip().str.lower()
    observacao = valores.map({"true": True, "false": False, "1": True, "0": False})
    if (valores.notna() & observacao.isna()).any():
        raise ValueError("Indicador de observacao invalido.")
    dados["has_valid_satellite_obs"] = observacao.astype("boolean")
    for coluna in NUMERICAS:
        if coluna in dados:
            dados[coluna] = pd.to_numeric(dados[coluna], errors="raise").replace([np.inf, -np.inf], np.nan)
    if not dados["grid_lat"].between(-90, 90).all() or not dados["grid_lon"].between(-180, 180).all():
        raise ValueError("Coordenadas invalidas ou ausentes.")
    if dados["hotspot_count"].isna().any() or (dados["hotspot_count"] < 0).any() or (dados["hotspot_count"] % 1 != 0).any():
        raise ValueError("A contagem de focos deve ser um inteiro nao negativo.")
    dados["country"] = dados["country"].str.strip().replace("", pd.NA)
    return dados.sort_values(["acq_date", "grid_id"]).reset_index(drop=True)


def caracteristicas(dados):
    x = dados.copy()
    dia = x["acq_date"].dt.dayofyear
    periodo = np.where(x["acq_date"].dt.is_leap_year, 366.0, 365.0)
    angulo = 2 * np.pi * (dia - 1) / periodo
    x["dia_seno"] = np.sin(angulo)
    x["dia_cosseno"] = np.cos(angulo)
    x["country"] = x["country"].fillna("__AUSENTE__").astype(str)
    return x[ENTRADAS]


def montar_base(dados):
    conhecida = dados["hotspot_count"].gt(0) | dados["has_valid_satellite_obs"].eq(True).fillna(False)
    atual = dados[conhecida].copy()
    resultado = dados[conhecida][["grid_id", "acq_date", "hotspot_count"]].copy()
    resultado["acq_date"] = resultado["acq_date"] - pd.Timedelta(days=1)
    resultado = resultado.rename(columns={"hotspot_count": "_focos_futuros"})
    base = atual.merge(resultado, on=["grid_id", "acq_date"], how="inner", validate="one_to_one")
    base[ALVO] = (base.pop("_focos_futuros") > 0).astype(int)
    if base.empty:
        raise ValueError("Nao ha pares consecutivos com deteccao ou observacao valida nos dois dias; nao e possivel treinar.")
    return base.sort_values(["acq_date", "grid_id"]).reset_index(drop=True)
