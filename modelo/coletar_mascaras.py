import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
from urllib.parse import urlparse

from dotenv import dotenv_values
import requests

from modelo.config import RAIZ

PASTA = RAIZ / "dados" / "evolucao"
CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"
AREA = "-47,-12,-43,-8"
TILES = {"h13v09", "h13v10"}
PADRAO = re.compile(r"^VJ114A1\.A(\d{7})\.(h\d{2}v\d{2})\.002\.\d{13}\.h5$")
MAGIC = b"\x89HDF\r\n\x1a\n"


def sha256(arquivo):
    h = hashlib.sha256()
    with Path(arquivo).open("rb") as entrada:
        for parte in iter(lambda: entrada.read(1024 * 1024), b""):
            h.update(parte)
    return h.hexdigest()


def substituir(origem, destino):
    for tentativa in range(7):
        try:
            os.replace(origem, destino)
            return
        except PermissionError:
            if tentativa == 6:
                raise
            time.sleep(min(5, 0.25 * 2 ** tentativa))


def salvar_manifesto(caminho, dados):
    temporario = caminho.with_suffix(".json.tmp")
    temporario.write_text(json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8")
    substituir(temporario, caminho)


def recuperar_arquivo(arquivo, item):
    from netCDF4 import Dataset

    with arquivo.open("rb") as entrada:
        if entrada.read(8) != MAGIC:
            raise ValueError("Arquivo recuperado nao e HDF5.")
    with Dataset(os.path.relpath(arquivo, Path.cwd())) as dados:
        if (dados.ShortName != "VJ114A1" or str(dados.VersionID) != "002" or
                dados.PlatformShortName != "JPSS-1" or dados.RangeBeginningDate != item["data"]):
            raise ValueError("Metadados do arquivo recuperado nao conferem.")
        grupo = dados.groups["HDFEOS"].groups["GRIDS"].groups["VIIRS_Grid_Daily_Fire"]
        mascara = grupo.groups["Data Fields"].variables["FireMask"][:]
        if mascara.shape != (1200, 1200):
            raise ValueError("Mascara recuperada tem dimensao incorreta.")
    return {**item, "sha256": sha256(arquivo), "bytes": arquivo.stat().st_size,
            "recuperado_em_utc": datetime.now(timezone.utc).isoformat()}


def meses(inicio, fim):
    atual = inicio
    while atual <= fim:
        seguinte = date(atual.year + (atual.month == 12), atual.month % 12 + 1, 1)
        ultimo = min(fim, seguinte - timedelta(days=1))
        yield atual, ultimo
        atual = seguinte


def descobrir(inicio, fim, pasta, area=AREA, tiles=None):
    tiles = TILES if tiles is None else set(tiles)
    catalogo = pasta / "brutos" / "catalogo"
    catalogo.mkdir(parents=True, exist_ok=True)
    arquivos = []
    for primeiro, ultimo in meses(inicio, fim):
        consulta = {
            "short_name": "VJ114A1", "version": "002", "bounding_box": area,
            "temporal": f"{primeiro.isoformat()}T00:00:00Z,{ultimo.isoformat()}T23:59:59Z",
            "page_size": 200, "sort_key": "start_date",
        }
        resposta = requests.get(CMR, params=consulta, timeout=60)
        resposta.raise_for_status()
        entradas = resposta.json()["feed"]["entry"]
        if int(resposta.headers.get("CMR-Hits", len(entradas))) != len(entradas):
            raise ValueError("A consulta CMR foi truncada; reduza o intervalo.")
        resumo = hashlib.sha256(resposta.content).hexdigest()
        (catalogo / f"{primeiro:%Y%m}_{resumo[:12]}.json").write_bytes(resposta.content)
        for item in entradas:
            titulo = item["title"]
            links = [x.get("href", "") for x in item.get("links", [])
                     if x.get("href", "").endswith(".h5") and
                     urlparse(x.get("href", "")).netloc == "data.lpdaac.earthdatacloud.nasa.gov"]
            if len(links) != 1:
                raise ValueError("Granulo com link ausente ou ambiguo.")
            url = links[0]
            nome = Path(urlparse(url).path).name
            match = PADRAO.fullmatch(nome)
            caminho = urlparse(url)
            if (not match or nome != titulo + ".h5" or caminho.scheme != "https" or
                    caminho.netloc != "data.lpdaac.earthdatacloud.nasa.gov" or
                    not caminho.path.startswith("/lp-prod-protected/VJ114A1.002/")):
                raise ValueError("Granulo ou endereco CMR inesperado.")
            if match.group(2) not in tiles:
                continue
            dia = datetime.strptime(match.group(1), "%Y%j").date()
            if not primeiro <= dia <= ultimo:
                raise ValueError("Data do nome do granulo fora da consulta.")
            arquivos.append({"nome": nome, "data": dia.isoformat(), "tile": match.group(2),
                             "url": url, "catalogo_sha256": resumo, "inicio_utc": item.get("time_start"),
                             "fim_utc": item.get("time_end"), "tamanho_cmr_mb": item.get("granule_size")})
    chaves = [(a["data"], a["tile"]) for a in arquivos]
    if len(chaves) != len(set(chaves)):
        raise ValueError("Mais de um granulo para a mesma data e tile; revise a proveniencia.")
    return sorted(arquivos, key=lambda a: (a["data"], a["tile"]))


def coletar(inicio, fim, pasta=PASTA, trabalhadores=2, area=AREA, tiles=None):
    inicio, fim, pasta = date.fromisoformat(inicio), date.fromisoformat(fim), Path(pasta)
    tiles = TILES if tiles is None else set(tiles)
    if not tiles or any(not re.fullmatch(r"h\d{2}v\d{2}", tile) for tile in tiles):
        raise ValueError("Tiles VIIRS invalidos.")
    if inicio > fim or fim >= datetime.now(timezone.utc).date():
        raise ValueError("Informe um intervalo historico terminado antes do dia UTC atual.")
    if not 1 <= trabalhadores <= 3:
        raise ValueError("Use de um a tres trabalhadores.")
    token = os.getenv("EARTHDATA_TOKEN") or dotenv_values(RAIZ / ".env").get("EARTHDATA_TOKEN")
    if not token:
        raise ValueError("Configure EARTHDATA_TOKEN no ambiente ou no .env local.")
    pasta_bruta = pasta / "brutos" / "viirs"
    pasta_bruta.mkdir(parents=True, exist_ok=True)
    manifesto_path = pasta / "manifesto_viirs.json"
    manifesto = json.loads(manifesto_path.read_text(encoding="utf-8")) if manifesto_path.exists() else {
        "produto": "VJ114A1.002", "area": area, "tiles": sorted(tiles), "arquivos": {}, "consultas": []}
    if manifesto.get("produto") != "VJ114A1.002" or manifesto.get("area") != area or manifesto.get("tiles", sorted(tiles)) != sorted(tiles):
        raise ValueError("Manifesto pertence a outra area ou produto.")
    arquivos = descobrir(inicio, fim, pasta, area, tiles)
    for nome, registro in manifesto["arquivos"].items():
        arquivo = pasta_bruta / nome
        if not arquivo.is_file() or sha256(arquivo) != registro["sha256"]:
            raise ValueError("Arquivo baixado anteriormente ausente ou com hash divergente.")
    recuperados = 0
    for item in arquivos:
        arquivo = pasta_bruta / item["nome"]
        if item["nome"] not in manifesto["arquivos"] and arquivo.is_file():
            manifesto["arquivos"][item["nome"]] = recuperar_arquivo(arquivo, item)
            recuperados += 1
    if recuperados:
        salvar_manifesto(manifesto_path, manifesto)
        print(f"Granulos recuperados e validados: {recuperados}", flush=True)
    pendentes = [a for a in arquivos if a["nome"] not in manifesto["arquivos"]]
    trava = threading.Lock()
    interromper = threading.Event()
    limite = {"ultimo": 0.0}

    def baixar(item):
        if interromper.is_set():
            return item, None, "interrompido apos falha de autenticacao"
        destino = pasta_bruta / item["nome"]
        temporario = destino.with_suffix(destino.suffix + ".part")
        for tentativa in range(3):
            with trava:
                atraso = 0.5 - (time.monotonic() - limite["ultimo"])
                if atraso > 0:
                    time.sleep(atraso)
                limite["ultimo"] = time.monotonic()
            try:
                with requests.get(item["url"], headers={"Authorization": "Bearer " + token},
                                  stream=True, timeout=60) as resposta:
                    if resposta.status_code in (401, 403):
                        interromper.set()
                        return item, None, f"autenticacao HTTP {resposta.status_code}"
                    if resposta.status_code == 429 or resposta.status_code >= 500:
                        if tentativa < 2:
                            time.sleep(min(30, 2 ** (tentativa + 1)))
                            continue
                        return item, None, f"servidor HTTP {resposta.status_code}"
                    if resposta.status_code != 200:
                        return item, None, f"indisponivel HTTP {resposta.status_code}"
                    tamanho = 0
                    esperado = int(resposta.headers.get("Content-Length", "0") or 0)
                    with temporario.open("wb") as arquivo:
                        for parte in resposta.iter_content(1024 * 256):
                            tamanho += len(parte)
                            if tamanho > 128 * 1024 * 1024:
                                raise ValueError("Granulo excede o limite de tamanho.")
                            arquivo.write(parte)
                    with temporario.open("rb") as arquivo:
                        valido = arquivo.read(8) == MAGIC
                    if not valido or (esperado and tamanho != esperado):
                        return item, None, "arquivo incompleto ou formato diferente de HDF5"
                digest = sha256(temporario)
                substituir(temporario, destino)
                return item, {**item, "sha256": digest, "bytes": tamanho,
                              "baixado_em_utc": datetime.now(timezone.utc).isoformat()}, None
            except (requests.RequestException, OSError, ValueError) as erro:
                if tentativa == 2:
                    return item, None, type(erro).__name__
                time.sleep(2 ** tentativa)
        return item, None, "falha inesperada"

    falhas = []
    with ThreadPoolExecutor(max_workers=trabalhadores) as executor:
        tarefas = [executor.submit(baixar, a) for a in pendentes]
        for numero, tarefa in enumerate(as_completed(tarefas), 1):
            item, registro, falha = tarefa.result()
            if falha:
                falhas.append({"nome": item["nome"], "motivo": falha})
            else:
                manifesto["arquivos"][item["nome"]] = registro
                salvar_manifesto(manifesto_path, manifesto)
            if numero % 10 == 0 or numero == len(pendentes):
                print(f"Granulos processados: {numero}/{len(pendentes)}; falhas: {len(falhas)}", flush=True)
    presentes = {(a["data"], a["tile"]) for a in arquivos if a["nome"] in manifesto["arquivos"]}
    total_dias = (fim - inicio).days + 1
    esperados = {(inicio + timedelta(days=i)).isoformat() for i in range(total_dias)}
    lacunas = [{"data": d, "tile": t} for d in sorted(esperados) for t in sorted(tiles) if (d, t) not in presentes]
    resultado = {"inicio": inicio.isoformat(), "fim": fim.isoformat(), "dias": total_dias,
                 "esperados": total_dias * len(tiles), "encontrados_catalogo": len(arquivos),
                 "baixados_ou_existentes": len(presentes), "lacunas": lacunas, "falhas": falhas,
                 "finalizado_em_utc": datetime.now(timezone.utc).isoformat()}
    manifesto["consultas"].append(resultado)
    salvar_manifesto(manifesto_path, manifesto)
    return resultado


def main():
    parser = argparse.ArgumentParser(description="Coleta mascaras diarias VIIRS para a area piloto.")
    parser.add_argument("--inicio", required=True)
    parser.add_argument("--fim", required=True)
    parser.add_argument("--pasta", type=Path, default=PASTA)
    parser.add_argument("--trabalhadores", type=int, default=2)
    args = parser.parse_args()
    resultado = coletar(args.inicio, args.fim, args.pasta, args.trabalhadores)
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
    if resultado["falhas"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
