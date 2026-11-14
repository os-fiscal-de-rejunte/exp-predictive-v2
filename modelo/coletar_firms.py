import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import threading
import time

from dotenv import dotenv_values
import numpy as np
import pandas as pd
import requests

from modelo.coletar_mascaras import PASTA, sha256
from modelo.config import RAIZ

API = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
AREA = "-47,-12,-43,-8"
FONTES = ["VIIRS_NOAA20_SP", "MODIS_SP"]
COLUNAS = ["latitude", "longitude", "acq_date", "acq_time", "satellite", "instrument", "confidence", "version", "frp"]


def segmentos(inicio, fim):
    atual = inicio
    while atual <= fim:
        dias = min(5, (fim - atual).days + 1)
        yield atual, dias
        atual += timedelta(days=dias)


def ler_csv(conteudo, fonte, inicio, dias, area=AREA):
    dados = pd.read_csv(io.BytesIO(conteudo), dtype={"acq_time": "string", "satellite": "string", "confidence": "string", "version": "string"})
    if not set(COLUNAS).issubset(dados.columns):
        raise ValueError("Resposta FIRMS sem as colunas esperadas.")
    dados = dados[COLUNAS].copy()
    if dados.empty:
        dados["instante_utc"] = pd.Series(dtype="datetime64[ns, UTC]")
        dados["fonte"] = pd.Series(dtype="string")
        return dados
    for coluna in ["latitude", "longitude", "frp"]:
        dados[coluna] = pd.to_numeric(dados[coluna], errors="raise")
    horas = dados["acq_time"].str.strip().str.zfill(4)
    dados["instante_utc"] = pd.to_datetime(dados["acq_date"].astype(str) + " " + horas, format="%Y-%m-%d %H%M", utc=True, errors="raise")
    limite_inferior = pd.Timestamp(inicio, tz="UTC")
    limite_superior = limite_inferior + pd.Timedelta(days=dias)
    oeste, sul, leste, norte = map(float, area.split(","))
    if (dados["latitude"].isna().any() or dados["longitude"].isna().any() or
            not dados["latitude"].between(sul, norte).all() or not dados["longitude"].between(oeste, leste).all() or
            not dados["instante_utc"].between(limite_inferior, limite_superior, inclusive="left").all()):
        raise ValueError("Data ou coordenada de deteccao fora da consulta.")
    if fonte.startswith("VIIRS_NOAA20") and not dados["satellite"].eq("N20").all():
        raise ValueError("Satelite inesperado para VIIRS NOAA-20.")
    dados["frp"] = dados["frp"].where(dados["frp"] >= 0, np.nan)
    dados["fonte"] = fonte
    return dados


def coletar(inicio, fim, pasta=PASTA, trabalhadores=3, fontes=None, area=AREA):
    inicio, fim, pasta = date.fromisoformat(inicio), date.fromisoformat(fim), Path(pasta)
    if inicio > fim or fim >= datetime.now(timezone.utc).date() or not 1 <= trabalhadores <= 4:
        raise ValueError("Periodo historico ou quantidade de trabalhadores invalida.")
    fontes_selecionadas = list(FONTES if fontes is None else fontes)
    permitidas = {"VIIRS_NOAA20_SP", "MODIS_SP", "VIIRS_NOAA20_NRT", "MODIS_NRT"}
    if not fontes_selecionadas or len(set(fontes_selecionadas)) != len(fontes_selecionadas) or not set(fontes_selecionadas).issubset(permitidas):
        raise ValueError("Fontes FIRMS invalidas ou duplicadas.")
    chave = os.getenv("FIRMS_MAP_KEY") or dotenv_values(RAIZ / ".env").get("FIRMS_MAP_KEY")
    if not chave:
        raise ValueError("Configure FIRMS_MAP_KEY no ambiente ou no .env local.")
    cache = pasta / "cache" / "firms"
    cache.mkdir(parents=True, exist_ok=True)
    trava = threading.Lock()
    interromper = threading.Event()
    limite = {"ultimo": 0.0}
    consultas = [(fonte, comeco, dias) for fonte in fontes_selecionadas for comeco, dias in segmentos(inicio, fim)]

    def buscar(consulta):
        fonte, comeco, dias = consulta
        arquivo = cache / f"{fonte}_{comeco:%Y%m%d}_{dias}.csv"
        recente = fonte.endswith("_NRT") and (datetime.now(timezone.utc).date() - fim).days <= 10
        novo = arquivo.exists() and (not recente or time.time() - arquivo.stat().st_mtime < 6 * 3600)
        if novo:
            conteudo = arquivo.read_bytes()
            try:
                dados = ler_csv(conteudo, fonte, comeco, dias, area)
                return consulta, dados, hashlib.sha256(conteudo).hexdigest(), None
            except (ValueError, KeyError, pd.errors.ParserError):
                return consulta, None, None, "cache invalido"
        if arquivo.exists():
            anterior = arquivo.read_bytes()
            digest_anterior = hashlib.sha256(anterior).hexdigest()
            copia = arquivo.with_name(arquivo.stem + "_" + digest_anterior[:16] + ".csv")
            if not copia.exists():
                copia.write_bytes(anterior)
        if interromper.is_set():
            return consulta, None, None, "interrompido apos falha de acesso"
        url = f"{API}/{chave}/{fonte}/{area}/{dias}/{comeco.isoformat()}"
        for tentativa in range(4):
            with trava:
                atraso = 0.35 - (time.monotonic() - limite["ultimo"])
                if atraso > 0:
                    time.sleep(atraso)
                limite["ultimo"] = time.monotonic()
            try:
                resposta = requests.get(url, timeout=90, allow_redirects=False)
                if resposta.status_code in (401, 403):
                    interromper.set()
                    return consulta, None, None, f"acesso HTTP {resposta.status_code}"
                if resposta.status_code == 429 or resposta.status_code >= 500:
                    if tentativa < 3:
                        time.sleep(min(30, 2 ** (tentativa + 1)))
                        continue
                    return consulta, None, None, f"HTTP {resposta.status_code}"
                if resposta.status_code != 200:
                    return consulta, None, None, f"HTTP {resposta.status_code}"
                dados = ler_csv(resposta.content, fonte, comeco, dias, area)
                temporario = arquivo.with_suffix(".csv.part")
                temporario.write_bytes(resposta.content)
                os.replace(temporario, arquivo)
                return consulta, dados, hashlib.sha256(resposta.content).hexdigest(), None
            except (requests.RequestException, OSError, ValueError, KeyError, pd.errors.ParserError) as erro:
                if tentativa == 3:
                    return consulta, None, None, type(erro).__name__
                time.sleep(2 ** tentativa)
        return consulta, None, None, "falha inesperada"

    partes, respostas, falhas = [], [], []
    with ThreadPoolExecutor(max_workers=trabalhadores) as executor:
        tarefas = [executor.submit(buscar, consulta) for consulta in consultas]
        for numero, tarefa in enumerate(as_completed(tarefas), 1):
            consulta, dados, digest, erro = tarefa.result()
            fonte, comeco, dias = consulta
            if erro:
                falhas.append({"fonte": fonte, "inicio": comeco.isoformat(), "dias": dias, "motivo": erro})
            else:
                partes.append(dados)
                arquivo_cache = cache / f"{fonte}_{comeco:%Y%m%d}_{dias}.csv"
                respostas.append({"fonte": fonte, "inicio": comeco.isoformat(), "dias": dias,
                                  "linhas": len(dados), "sha256_resposta": digest,
                                  "cache_gravado_utc": datetime.fromtimestamp(arquivo_cache.stat().st_mtime, timezone.utc).isoformat()})
            if numero % 20 == 0 or numero == len(consultas):
                print(f"Consultas FIRMS: {numero}/{len(consultas)}; falhas: {len(falhas)}", flush=True)
    if falhas:
        (pasta / "falhas_firms.json").write_text(json.dumps(falhas, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError("Consultas FIRMS incompletas; respostas validas permanecem no cache.")
    nao_vazios = [parte for parte in partes if not parte.empty]
    dados = pd.concat(nao_vazios, ignore_index=True) if nao_vazios else partes[0].copy()
    anteriores = len(dados)
    dados = dados.drop_duplicates().sort_values(["instante_utc", "fonte", "latitude", "longitude"]).reset_index(drop=True)
    caminho = pasta / f"firms_{inicio:%Y%m%d}_{fim:%Y%m%d}.parquet"
    dados.to_parquet(caminho, index=False)
    resultado = {"inicio": inicio.isoformat(), "fim": fim.isoformat(), "area": area,
                 "fontes": fontes_selecionadas, "consultas": len(consultas), "linhas": len(dados),
                 "duplicatas_exatas": anteriores - len(dados),
                 "por_fonte": dados["fonte"].value_counts().to_dict(), "sha256_arquivo": sha256(caminho),
                 "processado_em_utc": datetime.now(timezone.utc).isoformat(), "respostas": respostas,
                 "limite": "Sem registros nao comprova observacao nem ausencia de foco; arquivos SP podem diferir do NRT disponivel na epoca."}
    (pasta / f"manifesto_firms_{inicio:%Y%m%d}_{fim:%Y%m%d}.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    return resultado


def main():
    parser = argparse.ArgumentParser(description="Coleta historico FIRMS em intervalos de ate cinco dias.")
    parser.add_argument("--inicio", required=True)
    parser.add_argument("--fim", required=True)
    parser.add_argument("--pasta", type=Path, default=PASTA)
    parser.add_argument("--trabalhadores", type=int, default=3)
    parser.add_argument("--fontes", nargs="+")
    args = parser.parse_args()
    resultado = coletar(args.inicio, args.fim, args.pasta, args.trabalhadores, args.fontes)
    print(json.dumps({k: v for k, v in resultado.items() if k != "respostas"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
