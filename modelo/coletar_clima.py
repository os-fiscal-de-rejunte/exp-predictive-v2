import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd
import requests

from modelo.coletar_mascaras import PASTA, sha256
from modelo.grade import SAIDA as GRADE

URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
UNIDADES = {"T2M": "C", "RH2M": "%", "PRECTOTCORR": "mm/day", "WS10M": "m/s", "GWETROOT": "1"}


def celulas_meteorologicas(grade):
    dados = grade[["grid_id", "latitude", "longitude"]].copy()
    dados["lat_power"] = np.floor(dados["latitude"] / 0.5 + 0.5) * 0.5
    dados["lon_power"] = np.floor(dados["longitude"] / 0.625 + 0.5) * 0.625
    return dados


def validar_corpo(corpo, inicio, fim):
    header = corpo.get("header", {})
    if header.get("time_standard") != "UTC" or header.get("start") != inicio.strftime("%Y%m%d") or header.get("end") != fim.strftime("%Y%m%d"):
        raise ValueError("Periodo ou fuso incorreto no POWER.")
    for nome, unidade in UNIDADES.items():
        if corpo.get("parameters", {}).get(nome, {}).get("units") != unidade:
            raise ValueError(f"Unidade inesperada em {nome}.")
    valores = corpo.get("properties", {}).get("parameter", {})
    if set(valores) != set(UNIDADES):
        raise ValueError("Variaveis meteorologicas incompletas.")
    return valores, header.get("fill_value", -999)


def coletar(inicio, fim, pasta=PASTA, grade_path=GRADE, trabalhadores=3):
    inicio, fim, pasta = date.fromisoformat(inicio), date.fromisoformat(fim), Path(pasta)
    if inicio > fim or fim >= datetime.now(timezone.utc).date() or not 1 <= trabalhadores <= 4:
        raise ValueError("Periodo historico ou quantidade de trabalhadores invalida.")
    grade = celulas_meteorologicas(pd.read_csv(grade_path))
    locais = list(grade[["lat_power", "lon_power"]].drop_duplicates().itertuples(index=False, name=None))
    cache = pasta / "cache" / "power"
    cache.mkdir(parents=True, exist_ok=True)
    trava = threading.Lock()
    limite = {"ultimo": 0.0}

    def buscar(local):
        lat, lon = local
        chave = f"{lat:+07.3f}_{lon:+08.3f}_{inicio:%Y%m%d}_{fim:%Y%m%d}"
        arquivo = cache / (chave + ".json")
        atual = (datetime.now(timezone.utc).date() - fim).days <= 10
        novo = arquivo.exists() and (not atual or time.time() - arquivo.stat().st_mtime < 6 * 3600)
        if novo:
            conteudo = arquivo.read_bytes()
            corpo = json.loads(conteudo)
            validar_corpo(corpo, inicio, fim)
            return local, corpo, hashlib.sha256(conteudo).hexdigest(), None
        if arquivo.exists():
            anterior = arquivo.read_bytes()
            digest_anterior = hashlib.sha256(anterior).hexdigest()
            copia = arquivo.with_name(arquivo.stem + "_" + digest_anterior[:16] + ".json")
            if not copia.exists():
                copia.write_bytes(anterior)
        parametros = {"parameters": ",".join(UNIDADES), "community": "AG", "latitude": lat,
                      "longitude": lon, "start": inicio.strftime("%Y%m%d"), "end": fim.strftime("%Y%m%d"),
                      "format": "JSON", "time-standard": "UTC"}
        for tentativa in range(4):
            with trava:
                atraso = 0.5 - (time.monotonic() - limite["ultimo"])
                if atraso > 0:
                    time.sleep(atraso)
                limite["ultimo"] = time.monotonic()
            try:
                resposta = requests.get(URL, params=parametros, timeout=90)
                if resposta.status_code == 429 or resposta.status_code >= 500:
                    if tentativa < 3:
                        time.sleep(min(30, 2 ** (tentativa + 1)))
                        continue
                    return local, None, None, f"HTTP {resposta.status_code}"
                resposta.raise_for_status()
                corpo = resposta.json()
                validar_corpo(corpo, inicio, fim)
                coordenadas = corpo.get("geometry", {}).get("coordinates", [])
                if len(coordenadas) < 2 or abs(coordenadas[0] - lon) > 0.001 or abs(coordenadas[1] - lat) > 0.001:
                    raise ValueError("Coordenadas POWER divergentes.")
                temporario = arquivo.with_suffix(".json.part")
                temporario.write_bytes(resposta.content)
                os.replace(temporario, arquivo)
                return local, corpo, hashlib.sha256(resposta.content).hexdigest(), None
            except (requests.RequestException, OSError, ValueError, KeyError) as erro:
                if tentativa == 3:
                    return local, None, None, type(erro).__name__
                time.sleep(2 ** tentativa)
        return local, None, None, "falha inesperada"

    partes, fontes, falhas = [], [], []
    dias = pd.date_range(inicio, fim, freq="D")
    with ThreadPoolExecutor(max_workers=trabalhadores) as executor:
        tarefas = [executor.submit(buscar, local) for local in locais]
        for numero, tarefa in enumerate(as_completed(tarefas), 1):
            local, corpo, digest, erro = tarefa.result()
            if erro:
                falhas.append({"latitude": local[0], "longitude": local[1], "motivo": erro})
            else:
                valores, ausente = validar_corpo(corpo, inicio, fim)
                quadro = pd.DataFrame({"data_utc": dias, "lat_power": local[0], "lon_power": local[1]})
                for nome in UNIDADES:
                    v = valores[nome]
                    quadro[nome] = [np.nan if v.get(d.strftime("%Y%m%d")) in (None, ausente) else float(v[d.strftime("%Y%m%d")]) for d in dias]
                partes.append(quadro)
                chave = f"{local[0]:+07.3f}_{local[1]:+08.3f}_{inicio:%Y%m%d}_{fim:%Y%m%d}"
                arquivo_cache = cache / (chave + ".json")
                fontes.append({"latitude": local[0], "longitude": local[1], "sha256_resposta": digest,
                               "cache_gravado_utc": datetime.fromtimestamp(arquivo_cache.stat().st_mtime, timezone.utc).isoformat()})
            if numero % 5 == 0 or numero == len(locais):
                print(f"Celulas meteorologicas: {numero}/{len(locais)}; falhas: {len(falhas)}", flush=True)
    if falhas:
        caminho_falhas = pasta / "falhas_clima.json"
        caminho_falhas.write_text(json.dumps(falhas, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError("Consultas POWER incompletas; respostas validas foram preservadas no cache.")
    dados = pd.concat(partes, ignore_index=True).sort_values(["data_utc", "lat_power", "lon_power"])
    caminho = pasta / f"clima_{inicio:%Y%m%d}_{fim:%Y%m%d}.parquet"
    dados.to_parquet(caminho, index=False)
    resultado = {"fonte": URL, "inicio": inicio.isoformat(), "fim": fim.isoformat(), "celulas": len(locais),
                 "linhas": len(dados), "ausentes": dados[list(UNIDADES)].isna().sum().to_dict(),
                 "sha256_grade": sha256(grade_path), "sha256_arquivo": sha256(caminho),
                 "processado_em_utc": datetime.now(timezone.utc).isoformat(), "respostas": fontes,
                 "limite": "POWER historico pode ter sido revisado; nao prova disponibilidade na data simulada."}
    (pasta / f"manifesto_clima_{inicio:%Y%m%d}_{fim:%Y%m%d}.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    return resultado


def main():
    parser = argparse.ArgumentParser(description="Coleta clima historico POWER para a grade piloto.")
    parser.add_argument("--inicio", required=True)
    parser.add_argument("--fim", required=True)
    parser.add_argument("--pasta", type=Path, default=PASTA)
    parser.add_argument("--grade", type=Path, default=GRADE)
    parser.add_argument("--trabalhadores", type=int, default=3)
    args = parser.parse_args()
    resultado = coletar(args.inicio, args.fim, args.pasta, args.grade, args.trabalhadores)
    print(json.dumps({k: v for k, v in resultado.items() if k != "respostas"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
