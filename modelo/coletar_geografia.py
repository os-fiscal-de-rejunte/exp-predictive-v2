import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import threading
import time
from urllib.parse import urlparse

from dotenv import dotenv_values
import requests

from modelo.coletar_mascaras import AREA, CMR, PASTA, TILES, salvar_manifesto, sha256, substituir
from modelo.config import RAIZ

VEGETACAO = re.compile(r"^VJ113A2\.A(\d{7})\.(h\d{2}v\d{2})\.002\.\d{13}\.h5$")
RELEVO = re.compile(r"^NASADEM_HGT_s(09|10|11|12)w(044|045|046|047)\.zip$")


def catalogo(produto, inicio=None, fim=None, area=AREA, tiles=None, nomes_terreno=None):
    tiles = TILES if tiles is None else set(tiles)
    parametros = {"short_name": produto, "version": "002" if produto == "VJ113A2" else "001",
                  "bounding_box": area, "page_size": 200}
    if inicio and fim:
        parametros["temporal"] = f"{inicio.isoformat()}T00:00:00Z,{fim.isoformat()}T23:59:59Z"
    resposta = requests.get(CMR, params=parametros, timeout=60)
    resposta.raise_for_status()
    registros = resposta.json()["feed"]["entry"]
    if int(resposta.headers.get("CMR-Hits", len(registros))) != len(registros):
        raise ValueError("Catalogo geografico truncado.")
    saida = []
    for item in registros:
        links = [x.get("href", "") for x in item.get("links", [])
                 if urlparse(x.get("href", "")).netloc == "data.lpdaac.earthdatacloud.nasa.gov" and
                 x.get("href", "").endswith(".h5" if produto == "VJ113A2" else ".zip")]
        if len(links) != 1:
            raise ValueError("Endereco de dado geografico ausente ou ambiguo.")
        url = links[0]
        parsed = urlparse(url)
        nome = Path(parsed.path).name
        esperado = VEGETACAO.fullmatch(nome) if produto == "VJ113A2" else (nome in nomes_terreno if nomes_terreno is not None else RELEVO.fullmatch(nome))
        if not esperado:
            continue
        if parsed.scheme != "https" or not parsed.path.startswith(f"/lp-prod-protected/{produto}.{'002' if produto == 'VJ113A2' else '001'}/"):
            raise ValueError("Endereco geografico inesperado.")
        if produto == "VJ113A2":
            dia = datetime.strptime(esperado.group(1), "%Y%j").date()
            if esperado.group(2) not in tiles or dia.month != inicio.month or dia.year != inicio.year or dia.day > 8:
                continue
        saida.append({"nome": nome, "produto": produto, "url": url,
                      "inicio_utc": item.get("time_start"), "fim_utc": item.get("time_end"),
                      "tamanho_cmr_mb": float(item.get("granule_size", 0) or 0)})
    return saida


def descobrir(inicio, fim, area=AREA, tiles=None, nomes_terreno=None):
    registros = catalogo("NASADEM_HGT", area=area, tiles=tiles, nomes_terreno=nomes_terreno)
    atual = date(inicio.year, inicio.month, 1)
    while atual <= fim:
        ultimo = min(fim, atual + timedelta(days=7))
        registros.extend(catalogo("VJ113A2", atual, ultimo, area, tiles, nomes_terreno))
        atual = date(atual.year + (atual.month == 12), atual.month % 12 + 1, 1)
    nomes = [x["nome"] for x in registros]
    if len(nomes) != len(set(nomes)):
        raise ValueError("Granulos geograficos duplicados.")
    return sorted(registros, key=lambda x: (x["produto"], x["nome"]))


def coletar(inicio, fim, pasta=PASTA, amostra=False, trabalhadores=2, max_mb=3000, area=AREA, tiles=None, nomes_terreno=None):
    inicio, fim, pasta = date.fromisoformat(inicio), date.fromisoformat(fim), Path(pasta)
    if inicio > fim or fim >= datetime.now(timezone.utc).date() or not 1 <= trabalhadores <= 3 or max_mb <= 0:
        raise ValueError("Periodo ou limite de coleta invalido.")
    token = os.getenv("EARTHDATA_TOKEN") or dotenv_values(RAIZ / ".env").get("EARTHDATA_TOKEN")
    if not token:
        raise ValueError("Configure EARTHDATA_TOKEN no ambiente ou .env local.")
    registros = descobrir(inicio, fim, area, tiles, nomes_terreno)
    if amostra:
        registros = [next(x for x in registros if x["produto"] == nome) for nome in ["NASADEM_HGT", "VJ113A2"]]
    if sum(x["tamanho_cmr_mb"] for x in registros) > max_mb:
        raise ValueError("Coleta excede o limite de tamanho definido.")
    destino = pasta / "brutos" / "geografia"
    destino.mkdir(parents=True, exist_ok=True)
    manifesto_path = pasta / "manifesto_geografia.json"
    manifesto = json.loads(manifesto_path.read_text(encoding="utf-8")) if manifesto_path.exists() else {"arquivos": {}, "consultas": []}
    for nome, item in manifesto["arquivos"].items():
        if not (destino / nome).is_file() or sha256(destino / nome) != item["sha256"]:
            raise ValueError("Arquivo geografico anterior ausente ou alterado.")
    pendentes = [x for x in registros if x["nome"] not in manifesto["arquivos"]]
    trava = threading.Lock()
    parar = threading.Event()
    limite = {"ultimo": 0.0}

    def baixar(item):
        if parar.is_set():
            return item, None, "interrompido apos falha de autenticacao"
        caminho = destino / item["nome"]
        parcial = caminho.with_suffix(caminho.suffix + ".part")
        for tentativa in range(3):
            with trava:
                atraso = 0.5 - (time.monotonic() - limite["ultimo"])
                if atraso > 0:
                    time.sleep(atraso)
                limite["ultimo"] = time.monotonic()
            try:
                with requests.get(item["url"], headers={"Authorization": "Bearer " + token}, timeout=90, stream=True) as resposta:
                    if resposta.status_code in (401, 403):
                        parar.set()
                        return item, None, f"autenticacao HTTP {resposta.status_code}"
                    if resposta.status_code == 429 or resposta.status_code >= 500:
                        if tentativa < 2:
                            time.sleep(2 ** (tentativa + 1))
                            continue
                        return item, None, f"HTTP {resposta.status_code}"
                    if resposta.status_code != 200:
                        return item, None, f"HTTP {resposta.status_code}"
                    tamanho = 0
                    esperado = int(resposta.headers.get("Content-Length", 0) or 0)
                    with parcial.open("wb") as arquivo:
                        for bloco in resposta.iter_content(1024 * 256):
                            tamanho += len(bloco)
                            if tamanho > 128 * 1024 * 1024:
                                raise ValueError("Granulo excede o limite de tamanho.")
                            arquivo.write(bloco)
                    with parcial.open("rb") as arquivo:
                        assinatura = arquivo.read(8)
                    valido = assinatura == b"\x89HDF\r\n\x1a\n" if item["produto"] == "VJ113A2" else assinatura.startswith(b"PK\x03\x04")
                    if not valido or (esperado and tamanho != esperado):
                        return item, None, "arquivo incompleto ou formato inesperado"
                digest = sha256(parcial)
                substituir(parcial, caminho)
                return item, {**item, "sha256": digest, "bytes": tamanho,
                              "baixado_em_utc": datetime.now(timezone.utc).isoformat()}, None
            except (requests.RequestException, OSError, ValueError) as erro:
                if tentativa == 2:
                    return item, None, type(erro).__name__
                time.sleep(2 ** tentativa)
        return item, None, "falha inesperada"

    falhas = []
    with ThreadPoolExecutor(max_workers=trabalhadores) as executor:
        tarefas = [executor.submit(baixar, item) for item in pendentes]
        for numero, tarefa in enumerate(as_completed(tarefas), 1):
            item, registro, erro = tarefa.result()
            if erro:
                falhas.append({"nome": item["nome"], "motivo": erro})
            else:
                manifesto["arquivos"][item["nome"]] = registro
                salvar_manifesto(manifesto_path, manifesto)
            if numero % 5 == 0 or numero == len(pendentes):
                print(f"Granulos geograficos: {numero}/{len(pendentes)}; falhas: {len(falhas)}", flush=True)
    resultado = {"inicio": inicio.isoformat(), "fim": fim.isoformat(), "selecionados": len(registros),
                 "baixados_ou_existentes": sum(x["nome"] in manifesto["arquivos"] for x in registros),
                 "falhas": falhas, "amostra": amostra, "finalizado_em_utc": datetime.now(timezone.utc).isoformat()}
    manifesto["consultas"].append(resultado)
    salvar_manifesto(manifesto_path, manifesto)
    return resultado


def main():
    parser = argparse.ArgumentParser(description="Coleta vegetacao VIIRS e relevo NASADEM para o piloto.")
    parser.add_argument("--inicio", default="2023-12-01")
    parser.add_argument("--fim", default="2025-12-31")
    parser.add_argument("--pasta", type=Path, default=PASTA)
    parser.add_argument("--amostra", action="store_true")
    parser.add_argument("--trabalhadores", type=int, default=2)
    parser.add_argument("--max-mb", type=float, default=3000)
    args = parser.parse_args()
    resultado = coletar(args.inicio, args.fim, args.pasta, args.amostra, args.trabalhadores, args.max_mb)
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
    if resultado["falhas"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
