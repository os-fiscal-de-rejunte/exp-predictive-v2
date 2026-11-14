from datetime import timedelta
import hashlib
import io
import os
from pathlib import Path

from dotenv import dotenv_values
import numpy as np
import pandas as pd
import requests

from modelo.coletar_firms import API, ler_csv, segmentos
from modelo.coletar_mascaras import PASTA
from modelo.config import RAIZ

RAIO = 6371007.181
FONTES = {"viirs": "VIIRS_NOAA20_NRT", "modis": "MODIS_NRT"}


def limites(geometria):
    x0, y0 = float(geometria["xllcorner"]), float(geometria["yllcorner"])
    x1 = x0 + int(geometria["ncols"]) * float(geometria["cellsize_m"])
    y1 = y0 + int(geometria["nrows"]) * float(geometria["cellsize_m"])
    latitudes = np.rad2deg(np.array([y0, y1]) / RAIO)
    longitudes = [np.rad2deg(x / (RAIO * np.cos(np.deg2rad(y)))) for x in [x0, x1] for y in latitudes]
    return min(longitudes) - 0.001, min(latitudes) - 0.001, max(longitudes) + 0.001, max(latitudes) + 0.001


def dentro(dados, geometria):
    lat = pd.to_numeric(dados["latitude"], errors="raise")
    lon = pd.to_numeric(dados["longitude"], errors="raise")
    y = RAIO * np.deg2rad(lat)
    x = RAIO * np.deg2rad(lon) * np.cos(np.deg2rad(lat))
    x0, y0 = float(geometria["xllcorner"]), float(geometria["yllcorner"])
    x1 = x0 + int(geometria["ncols"]) * float(geometria["cellsize_m"])
    y1 = y0 + int(geometria["nrows"]) * float(geometria["cellsize_m"])
    return dados[(x >= x0) & (x < x1) & (y >= y0) & (y < y1)].copy()


def disponibilidade(chave, fonte):
    url = f"https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/{chave}/{fonte}"
    try:
        resposta = requests.get(url, timeout=30, allow_redirects=False)
        if resposta.status_code != 200:
            raise RuntimeError(f"FIRMS indisponivel (HTTP {resposta.status_code}).")
        dados = pd.read_csv(io.BytesIO(resposta.content))
        linha = dados[dados["data_id"].eq(fonte)]
        if len(linha) != 1:
            raise RuntimeError("Disponibilidade da fonte FIRMS nao confirmada.")
        return pd.Timestamp(linha.iloc[0]["min_date"]).date(), pd.Timestamp(linha.iloc[0]["max_date"]).date()
    except (requests.RequestException, KeyError, ValueError, pd.errors.ParserError):
        raise RuntimeError("Nao foi possivel consultar a disponibilidade FIRMS.") from None


def consultar(geometria, data_alvo, fontes=FONTES, pasta=PASTA):
    chave = os.getenv("FIRMS_MAP_KEY") or dotenv_values(RAIZ / ".env").get("FIRMS_MAP_KEY")
    if not chave:
        raise RuntimeError("FIRMS_MAP_KEY nao configurada.")
    inicio, fim = data_alvo - timedelta(days=9), data_alvo - timedelta(days=2)
    caixa = ",".join(f"{v:.6f}" for v in limites(geometria))
    cache = Path(pasta) / "cache" / "firms_atual"
    cache.mkdir(parents=True, exist_ok=True)
    resultado = {}
    for nome, fonte in fontes.items():
        try:
            minimo, maximo = disponibilidade(chave, fonte)
            if inicio < minimo or fim > maximo:
                raise RuntimeError("Janela solicitada fora do intervalo disponivel da fonte.")
            partes, hashes = [], []
            for primeiro, dias in segmentos(inicio, fim):
                url = f"{API}/{chave}/{fonte}/{caixa}/{dias}/{primeiro.isoformat()}"
                resposta = requests.get(url, timeout=60, allow_redirects=False)
                if resposta.status_code != 200:
                    raise RuntimeError(f"Consulta FIRMS recusada (HTTP {resposta.status_code}).")
                dados = ler_csv(resposta.content, fonte, primeiro, dias, caixa)
                digest = hashlib.sha256(resposta.content).hexdigest()
                arquivo = cache / f"{fonte}_{primeiro:%Y%m%d}_{dias}_{digest[:16]}.csv"
                if not arquivo.exists():
                    arquivo.write_bytes(resposta.content)
                partes.append(dentro(dados, geometria))
                hashes.append(digest)
            nao_vazias = [p for p in partes if not p.empty]
            dados = pd.concat(nao_vazias, ignore_index=True).drop_duplicates() if nao_vazias else partes[0]
            registros = len(dados)
            frp = dados["frp"].sum(min_count=1) if registros else np.nan
            resultado[nome] = {"situacao": "consultado", "registros_positivos": registros if registros else None,
                               "frp_mw_soma_registros": float(frp) if pd.notna(frp) else None,
                               "inicio_utc": inicio.isoformat(), "fim_utc": fim.isoformat(),
                               "sha256_respostas": hashes,
                               "cobertura_observacional": "nao comprovada; ausencia de registros permanece desconhecida"}
        except (requests.RequestException, RuntimeError, ValueError, KeyError, OSError, pd.errors.ParserError) as erro:
            resultado[nome] = {"situacao": "indisponivel", "registros_positivos": None,
                               "motivo": str(erro) if isinstance(erro, RuntimeError) else type(erro).__name__}
    return resultado
