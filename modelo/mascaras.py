import argparse
from datetime import date, timedelta
import json
import os
from pathlib import Path

from netCDF4 import Dataset
import numpy as np
import pandas as pd

from modelo.coletar_mascaras import PASTA, TILES, sha256
from modelo.grade import SAIDA as GRADE

RAIO = 6371007.181
VALIDAS = np.array([3, 5, 7, 8, 9])
FOGO = np.array([7, 8, 9])


def abrir(caminho):
    return Dataset(os.path.relpath(caminho, Path.cwd()))


def geometria(arquivo, grade):
    with abrir(arquivo) as dados:
        if dados.ShortName != "VJ114A1" or str(dados.VersionID) != "002" or dados.PlatformShortName != "JPSS-1":
            raise ValueError("Produto, versao ou plataforma incorretos.")
        grupo = dados.groups["HDFEOS"].groups["GRIDS"].groups["VIIRS_Grid_Daily_Fire"]
        campos = grupo.groups["Data Fields"]
        projecao = campos.variables["Projection"]
        if projecao.grid_mapping_name != "sinusoidal" or abs(float(projecao.earth_radius) - RAIO) > 0.01:
            raise ValueError("Projecao VIIRS inesperada.")
        x = np.asarray(grupo.variables["XDim"][:], dtype=float)
        y = np.asarray(grupo.variables["YDim"][:], dtype=float)
        if len(x) != 1200 or len(y) != 1200 or not np.allclose(np.diff(x), np.diff(x)[0], atol=0.02) or not np.allclose(np.diff(y), np.diff(y)[0], atol=0.02):
            raise ValueError("Grade do granulo VIIRS inesperada.")
        if np.diff(x)[0] <= 0 or np.diff(y)[0] >= 0:
            raise ValueError("Orientacao da grade VIIRS inesperada.")
        x = x + np.diff(x)[0] / 2
        y = y + np.diff(y)[0] / 2
        return x, y


def mapa_pixels(x, y, grade):
    if len(grade) != 256 or not grade["fracao_brasil_aprox"].ge(0.999999).all():
        raise ValueError("Este piloto exige a grade completa e inteiramente brasileira validada.")
    latitudes = np.rad2deg(y / RAIO)
    linhas_uteis = np.flatnonzero((latitudes >= -12) & (latitudes < -8))
    lat = latitudes[linhas_uteis, None]
    lon = np.rad2deg(x[None, :] / (RAIO * np.cos(np.deg2rad(lat))))
    dentro = (lon >= -47) & (lon < -43)
    r, c = np.nonzero(dentro)
    lat_pixel = latitudes[linhas_uteis[r]]
    lon_pixel = lon[r, c]
    linha = np.floor((lat_pixel + 12) / 0.25).astype(int)
    coluna = np.floor((lon_pixel + 47) / 0.25).astype(int)
    identificador = linha * 16 + coluna
    return linhas_uteis[r], c, identificador


def extrair(arquivo, data, mapa, coordenadas):
    with abrir(arquivo) as dados:
        if (dados.ShortName != "VJ114A1" or str(dados.VersionID) != "002" or
                dados.PlatformShortName != "JPSS-1" or dados.RangeBeginningDate != data):
            raise ValueError("Metadados do granulo nao conferem com o manifesto.")
        grupo = dados.groups["HDFEOS"].groups["GRIDS"].groups["VIIRS_Grid_Daily_Fire"]
        x = np.asarray(grupo.variables["XDim"][:], dtype=float)
        y = np.asarray(grupo.variables["YDim"][:], dtype=float)
        if not np.allclose(x[[0, -1]], coordenadas[0], atol=0.02) or not np.allclose(y[[0, -1]], coordenadas[1], atol=0.02):
            raise ValueError("A grade espacial mudou entre granulos.")
        mascara = grupo.groups["Data Fields"].variables["FireMask"][:]
        valores = np.ma.filled(np.ma.asarray(mascara, dtype=np.int16), -1)
        linhas, colunas, ids = mapa
        amostra = valores[linhas, colunas]
        contagens = np.zeros((256, 10), dtype=np.int64)
        for classe in range(10):
            contagens[:, classe] = np.bincount(ids[amostra == classe], minlength=256)
        cobertos = np.bincount(ids, minlength=256)
        return contagens, cobertos, str(dados.ProductionTime)


def processar(inicio, fim, pasta=PASTA, grade_path=GRADE):
    inicio, fim, pasta = date.fromisoformat(inicio), date.fromisoformat(fim), Path(pasta)
    if inicio > fim:
        raise ValueError("Periodo invalido.")
    grade = pd.read_csv(grade_path).sort_values(["linha", "coluna"]).reset_index(drop=True)
    if grade.duplicated(["linha", "coluna"]).any() or not np.array_equal(grade["linha"] * 16 + grade["coluna"], np.arange(256)):
        raise ValueError("Grade invalida ou com celulas ausentes.")
    manifesto_path = pasta / "manifesto_viirs.json"
    manifesto = json.loads(manifesto_path.read_text(encoding="utf-8"))
    brutos = pasta / "brutos" / "viirs"
    por_dia = {}
    for item in manifesto["arquivos"].values():
        if inicio.isoformat() <= item["data"] <= fim.isoformat():
            por_dia.setdefault(item["data"], {})[item["tile"]] = item
    mapas, coordenadas, esperados = {}, {}, np.zeros(256, dtype=np.int64)
    for tile in sorted(TILES):
        primeiro = next((v[tile] for _, v in sorted(por_dia.items()) if tile in v), None)
        if primeiro is None:
            raise ValueError(f"Sem granulo de referencia para {tile}.")
        arquivo = brutos / primeiro["nome"]
        if sha256(arquivo) != primeiro["sha256"]:
            raise ValueError("Granulo de referencia alterado.")
        with abrir(arquivo) as dados:
            grupo = dados.groups["HDFEOS"].groups["GRIDS"].groups["VIIRS_Grid_Daily_Fire"]
            coordenadas[tile] = (np.asarray(grupo.variables["XDim"][:])[[0, -1]], np.asarray(grupo.variables["YDim"][:])[[0, -1]])
        x, y = geometria(arquivo, grade)
        mapas[tile] = mapa_pixels(x, y, grade)
        esperados += np.bincount(mapas[tile][2], minlength=256)
    if (esperados == 0).any():
        raise ValueError("Existem celulas sem pixels de referencia.")
    linhas = []
    fontes = {}
    for dia in range((fim - inicio).days + 1):
        data = (inicio + timedelta(days=dia)).isoformat()
        totais = np.zeros((256, 10), dtype=np.int64)
        cobertos = np.zeros(256, dtype=np.int64)
        producoes = []
        fontes[data] = []
        for tile, item in sorted(por_dia.get(data, {}).items()):
            arquivo = brutos / item["nome"]
            if sha256(arquivo) != item["sha256"]:
                raise ValueError(f"Granulo alterado: {item['nome']}")
            classes, pixels, producao = extrair(arquivo, data, mapas[tile], coordenadas[tile])
            totais += classes
            cobertos += pixels
            producoes.append(producao)
            fontes[data].append({"tile": tile, "arquivo": item["nome"], "sha256": item["sha256"], "producao_utc": producao})
        if (cobertos > esperados).any():
            raise ValueError("Pixels de tiles sobrepostos.")
        validos = totais[:, VALIDAS].sum(axis=1)
        focos = totais[:, FOGO].sum(axis=1)
        completos = validos == esperados
        alvo = pd.array(np.where(completos, (focos > 0).astype(float), np.nan), dtype="Int64")
        quadro = grade[["grid_id", "pais", "linha", "coluna", "latitude", "longitude"]].copy()
        quadro["data_utc"] = data
        quadro["pixels_esperados"] = esperados
        quadro["pixels_cobertos"] = cobertos
        quadro["pixels_validos"] = validos
        quadro["pixels_fogo"] = focos
        quadro["pixels_nuvem"] = totais[:, 4]
        quadro["pixels_desconhecidos"] = esperados - validos - totais[:, 4]
        quadro["mascara_completa"] = completos
        quadro["evidencia_foco"] = focos > 0
        quadro["alvo_condicional"] = alvo
        quadro["producao_utc"] = max(producoes) if len(producoes) == len(TILES) else None
        linhas.append(quadro)
        if (dia + 1) % 30 == 0 or dia + 1 == (fim - inicio).days + 1:
            print(f"Dias processados: {dia + 1}/{(fim - inicio).days + 1}", flush=True)
    resultado = pd.concat(linhas, ignore_index=True)
    caminho = pasta / f"mascaras_{inicio:%Y%m%d}_{fim:%Y%m%d}.parquet"
    resultado.to_parquet(caminho, index=False)
    fontes_path = pasta / f"fontes_mascaras_{inicio:%Y%m%d}_{fim:%Y%m%d}.json"
    fontes_path.write_text(json.dumps(fontes, ensure_ascii=False, indent=2), encoding="utf-8")
    completo = resultado["mascara_completa"]
    resumo = {
        "inicio": inicio.isoformat(), "fim": fim.isoformat(), "linhas": len(resultado),
        "celulas": len(grade), "completas": int(completo.sum()),
        "positivas_completas": int((completo & resultado["evidencia_foco"]).sum()),
        "negativas_completas": int((completo & ~resultado["evidencia_foco"]).sum()),
        "incompletas": int((~completo).sum()),
        "positivas_incompletas": int((~completo & resultado["evidencia_foco"]).sum()),
        "sha256_grade": sha256(grade_path), "sha256_resultado": sha256(caminho),
        "arquivo": caminho.name, "fontes": fontes_path.name,
    }
    (pasta / f"resumo_mascaras_{inicio:%Y%m%d}_{fim:%Y%m%d}.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    return resumo


def main():
    parser = argparse.ArgumentParser(description="Agrega mascaras VIIRS em celulas e dias.")
    parser.add_argument("--inicio", required=True)
    parser.add_argument("--fim", required=True)
    parser.add_argument("--pasta", type=Path, default=PASTA)
    parser.add_argument("--grade", type=Path, default=GRADE)
    args = parser.parse_args()
    print(json.dumps(processar(args.inicio, args.fim, args.pasta, args.grade), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
