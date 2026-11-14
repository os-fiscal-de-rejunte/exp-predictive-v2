import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
from urllib.parse import urlparse

from dotenv import dotenv_values
from pyhdf.SD import SD, SDC
import requests

from modelo.coletar_mascaras import AREA, CMR, PASTA, TILES, salvar_manifesto, sha256, substituir
from modelo.config import RAIZ

PADRAO = re.compile(r"^MCD12Q1\.A(\d{4})001\.(h\d{2}v\d{2})\.061\.(\d{13})\.hdf$")


def descobrir(inicio, fim):
    parametros = {"short_name": "MCD12Q1", "version": "061", "bounding_box": AREA, "page_size": 100,
                  "temporal": f"{inicio}-01-01T00:00:00Z,{fim}-12-31T23:59:59Z"}
    resposta = requests.get(CMR, params=parametros, timeout=60)
    resposta.raise_for_status()
    entradas = resposta.json()["feed"]["entry"]
    if int(resposta.headers.get("CMR-Hits", len(entradas))) != len(entradas):
        raise ValueError("Catalogo de cobertura do solo truncado.")
    origem = hashlib.sha256(resposta.content).hexdigest()
    arquivos = []
    for entrada in entradas:
        links = [x.get("href", "") for x in entrada.get("links", [])
                 if x.get("href", "").endswith(".hdf") and
                 urlparse(x.get("href", "")).netloc == "data.lpdaac.earthdatacloud.nasa.gov"]
        if len(links) != 1:
            raise ValueError("Link de cobertura do solo ausente ou ambiguo.")
        url = links[0]
        caminho = urlparse(url)
        nome = Path(caminho.path).name
        partes = PADRAO.fullmatch(nome)
        if not partes or caminho.scheme != "https" or not caminho.path.startswith("/lp-prod-protected/MCD12Q1.061/"):
            raise ValueError("Granulo de cobertura do solo inesperado.")
        ano, tile, producao = partes.groups()
        if tile not in TILES or not inicio <= int(ano) <= fim:
            continue
        arquivos.append({"nome": nome, "ano": int(ano), "tile": tile, "url": url,
                         "producao_utc": datetime.strptime(producao, "%Y%j%H%M%S").replace(tzinfo=timezone.utc).isoformat(),
                         "catalogo_atualizado_utc": entrada["updated"], "catalogo_sha256": origem})
    chaves = {(x["ano"], x["tile"]) for x in arquivos}
    if len(chaves) != len(arquivos):
        raise ValueError("Granulos de cobertura do solo duplicados.")
    return sorted(arquivos, key=lambda x: (x["ano"], x["tile"]))


def verificar_hdf(caminho):
    arquivo = SD(os.path.relpath(caminho, Path.cwd()), SDC.READ)
    try:
        if not {"LC_Type1", "QC"}.issubset(arquivo.datasets()):
            raise ValueError("Camadas esperadas de cobertura do solo ausentes.")
        if arquivo.select("LC_Type1").info()[2] != [2400, 2400]:
            raise ValueError("Resolucao de cobertura do solo inesperada.")
    finally:
        arquivo.end()


def coletar(inicio=2022, fim=2024, pasta=PASTA):
    if inicio > fim or fim >= datetime.now(timezone.utc).year:
        raise ValueError("Anos de cobertura do solo invalidos.")
    pasta = Path(pasta)
    token = os.getenv("EARTHDATA_TOKEN") or dotenv_values(RAIZ / ".env").get("EARTHDATA_TOKEN")
    if not token:
        raise ValueError("Configure EARTHDATA_TOKEN no ambiente ou .env local.")
    destino = pasta / "brutos" / "cobertura_solo"
    destino.mkdir(parents=True, exist_ok=True)
    manifesto_path = pasta / "manifesto_cobertura_solo.json"
    manifesto = json.loads(manifesto_path.read_text(encoding="utf-8")) if manifesto_path.exists() else {"produto": "MCD12Q1.061", "arquivos": {}}
    if manifesto["produto"] != "MCD12Q1.061":
        raise ValueError("Manifesto pertence a outro produto.")
    arquivos = descobrir(inicio, fim)
    for numero, item in enumerate(arquivos, 1):
        caminho = destino / item["nome"]
        anterior = manifesto["arquivos"].get(item["nome"])
        if anterior:
            if not caminho.is_file() or sha256(caminho) != anterior["sha256"]:
                raise ValueError("Granulo anterior de cobertura do solo alterado.")
        else:
            parcial = caminho.with_suffix(".part")
            for tentativa in range(3):
                try:
                    with requests.get(item["url"], headers={"Authorization": "Bearer " + token}, stream=True, timeout=90) as resposta:
                        if resposta.status_code in (401, 403):
                            raise ValueError(f"Autenticacao Earthdata HTTP {resposta.status_code}.")
                        resposta.raise_for_status()
                        tamanho = 0
                        with parcial.open("wb") as arquivo:
                            for bloco in resposta.iter_content(1024 * 256):
                                tamanho += len(bloco)
                                if tamanho > 100 * 1024 * 1024:
                                    raise ValueError("Granulo excede o tamanho permitido.")
                                arquivo.write(bloco)
                        esperado = int(resposta.headers.get("Content-Length", "0") or 0)
                        if esperado and tamanho != esperado:
                            raise ValueError("Granulo incompleto.")
                    verificar_hdf(parcial)
                    substituir(parcial, caminho)
                    manifesto["arquivos"][item["nome"]] = {**item, "sha256": sha256(caminho),
                                                             "baixado_em_utc": datetime.now(timezone.utc).isoformat()}
                    salvar_manifesto(manifesto_path, manifesto)
                    break
                except requests.RequestException as erro:
                    if tentativa == 2:
                        raise RuntimeError(f"Falha na coleta de cobertura do solo: {type(erro).__name__}.") from None
                    time.sleep(2 ** tentativa)
        print(f"Cobertura do solo: {numero}/{len(arquivos)}", flush=True)
    esperados = {(ano, tile) for ano in range(inicio, fim + 1) for tile in TILES}
    encontrados = {(x["ano"], x["tile"]) for x in arquivos}
    return {"inicio": inicio, "fim": fim, "encontrados": len(encontrados),
            "ausentes": [{"ano": a, "tile": t} for a, t in sorted(esperados - encontrados)]}


def main():
    parser = argparse.ArgumentParser(description="Coleta cobertura do solo anual MODIS na area piloto.")
    parser.add_argument("--inicio", type=int, default=2022)
    parser.add_argument("--fim", type=int, default=2024)
    parser.add_argument("--pasta", type=Path, default=PASTA)
    args = parser.parse_args()
    print(json.dumps(coletar(args.inicio, args.fim, args.pasta), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
