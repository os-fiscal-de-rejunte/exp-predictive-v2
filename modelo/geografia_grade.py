import argparse
import json
import os
from pathlib import Path
import re
from zipfile import ZipFile

from netCDF4 import Dataset
import numpy as np
import pandas as pd

from modelo.coletar_mascaras import PASTA, TILES, sha256
from modelo.grade import SAIDA as GRADE
from modelo.mascaras import RAIO, mapa_pixels

TILE_TERRENO = re.compile(r"^NASADEM_HGT_s(\d{2})w(\d{3})\.zip$")


def ler_manifesto(pasta):
    pasta = Path(pasta)
    manifesto = json.loads((pasta / "manifesto_geografia.json").read_text(encoding="utf-8"))
    origem = pasta / "brutos" / "geografia"
    for nome, item in manifesto["arquivos"].items():
        arquivo = origem / nome
        if not arquivo.is_file() or sha256(arquivo) != item["sha256"]:
            raise ValueError("Granulo geografico ausente ou alterado.")
    return manifesto, origem


def relevo(manifesto, origem, grade):
    itens = [x for x in manifesto["arquivos"].values() if x["produto"] == "NASADEM_HGT"]
    if len(itens) != 16:
        raise ValueError(f"Eram esperados 16 tiles de relevo; encontrados {len(itens)}.")
    somas, quadrados, contagens = np.zeros(256), np.zeros(256), np.zeros(256, dtype=np.int64)
    for item in itens:
        match = TILE_TERRENO.fullmatch(item["nome"])
        if not match:
            raise ValueError("Nome do tile NASADEM inesperado.")
        sul, oeste = -int(match.group(1)), -int(match.group(2))
        with ZipFile(origem / item["nome"]) as arquivo:
            membros = [n for n in arquivo.namelist() if n.lower().endswith(".hgt")]
            if len(membros) != 1:
                raise ValueError("Arquivo NASADEM sem raster unico.")
            conteudo = arquivo.read(membros[0])
        if len(conteudo) != 3601 * 3601 * 2:
            raise ValueError("Dimensao do raster NASADEM inesperada.")
        altura = np.frombuffer(conteudo, dtype=">i2").reshape(3601, 3601)[:3600, :3600]
        latitudes = sul + 1 - np.arange(3600) / 3600
        longitudes = oeste + np.arange(3600) / 3600
        linhas = np.floor((latitudes + 12) / 0.25).astype(int)
        colunas = np.floor((longitudes + 47) / 0.25).astype(int)
        for linha in np.unique(linhas[(linhas >= 0) & (linhas < 16)]):
            l = np.flatnonzero(linhas == linha)
            for coluna in np.unique(colunas[(colunas >= 0) & (colunas < 16)]):
                c = np.flatnonzero(colunas == coluna)
                bloco = altura[l[0]:l[-1] + 1, c[0]:c[-1] + 1]
                valido = bloco[(bloco >= -500) & (bloco <= 9000)].astype(np.float64)
                indice = linha * 16 + coluna
                somas[indice] += valido.sum()
                quadrados[indice] += np.square(valido).sum()
                contagens[indice] += len(valido)
    media = np.divide(somas, contagens, out=np.full(256, np.nan), where=contagens > 0)
    variancia = np.divide(quadrados, contagens, out=np.full(256, np.nan), where=contagens > 0) - media ** 2
    return pd.DataFrame({"grid_id": grade["grid_id"], "altitude_media_m": media,
                         "altitude_desvio_m": np.sqrt(np.maximum(0, variancia)), "altitude_pixels_validos": contagens})


def vegetar(arquivo, grade):
    with Dataset(os.path.relpath(arquivo, Path.cwd())) as dados:
        if dados.ShortName != "VJ113A2" or str(dados.VersionID) != "002" or dados.PlatformShortName != "JPSS-1":
            raise ValueError("Produto de vegetacao inesperado.")
        grupo = dados.groups["HDFEOS"].groups["GRIDS"].groups["VIIRS_Grid_16Day_VI_1km"]
        campos = grupo.groups["Data Fields"]
        projecao = campos.variables["Projection"]
        if projecao.grid_mapping_name != "sinusoidal" or abs(float(projecao.earth_radius) - RAIO) > 0.01:
            raise ValueError("Projecao da vegetacao inesperada.")
        x = np.asarray(grupo.variables["XDim"][:], dtype=float)
        y = np.asarray(grupo.variables["YDim"][:], dtype=float)
        if len(x) != 1200 or len(y) != 1200:
            raise ValueError("Dimensao de vegetacao inesperada.")
        x = x + np.diff(x)[0] / 2
        y = y + np.diff(y)[0] / 2
        linhas, colunas, ids = mapa_pixels(x, y, grade)
        arrays = []
        for nome in ["1 km 16 days NDVI", "1 km 16 days EVI", "1 km 16 days VI Quality", "1 km 16 days pixel reliability"]:
            var = campos.variables[nome]
            var.set_auto_maskandscale(False)
            arrays.append(np.asarray(var[:])[linhas, colunas])
        ndvi, evi, qa, confiabilidade = arrays
        bom = ((qa & 3) == 0) & ((qa & (1 << 8)) == 0) & ((qa & (1 << 10)) == 0)
        bom &= ((qa & (3 << 14)) == 0) & (((qa >> 11) & 7) == 1)
        bom &= (confiabilidade >= 0) & (confiabilidade <= 2)
        bom &= (ndvi >= -10000) & (ndvi <= 10000) & (evi >= -10000) & (evi <= 10000)
        n = np.bincount(ids[bom], minlength=256)
        soma_ndvi = np.bincount(ids[bom], weights=ndvi[bom].astype(float) * 0.0001, minlength=256)
        soma_evi = np.bincount(ids[bom], weights=evi[bom].astype(float) * 0.0001, minlength=256)
        esperados = np.bincount(ids, minlength=256)
        return {"soma_ndvi": soma_ndvi, "soma_evi": soma_evi, "validos": n, "esperados": esperados,
                "inicio": str(dados.RangeBeginningDate), "fim": str(dados.RangeEndingDate),
                "producao": str(dados.ProductionTime)}


def vegetacao(manifesto, origem, grade):
    itens = [x for x in manifesto["arquivos"].values() if x["produto"] == "VJ113A2"]
    por_inicio = {}
    for item in itens:
        por_inicio.setdefault(item["nome"].split(".")[1], []).append(item)
    quadros = []
    for numero, (nome, arquivos) in enumerate(sorted(por_inicio.items()), 1):
        acumulado = {"soma_ndvi": np.zeros(256), "soma_evi": np.zeros(256),
                     "validos": np.zeros(256, dtype=int), "esperados": np.zeros(256, dtype=int)}
        fins, producoes, inicios, tiles = [], [], [], set()
        for item in arquivos:
            tile = item["nome"].split(".")[2]
            if tile not in TILES or tile in tiles:
                raise ValueError("Tile de vegetacao duplicado ou inesperado.")
            tiles.add(tile)
            parte = vegetar(origem / item["nome"], grade)
            for chave in acumulado:
                acumulado[chave] += parte[chave]
            inicios.append(parte["inicio"])
            fins.append(parte["fim"])
            producoes.append(parte["producao"])
        if len(set(inicios)) != 1 or len(set(fins)) != 1:
            raise ValueError("Datas do composto de vegetacao inconsistentes.")
        esperado = acumulado["esperados"]
        frac = np.divide(acumulado["validos"], esperado, out=np.zeros(256), where=esperado > 0)
        confiavel = (frac >= 0.2) & (len(tiles) == len(TILES))
        n = acumulado["validos"]
        ndvi = np.divide(acumulado["soma_ndvi"], n, out=np.full(256, np.nan), where=(n > 0) & confiavel)
        evi = np.divide(acumulado["soma_evi"], n, out=np.full(256, np.nan), where=(n > 0) & confiavel)
        disponivel = max(pd.Timestamp(max(producoes), tz="UTC"), pd.Timestamp(max(fins), tz="UTC") + pd.Timedelta(days=1))
        quadros.append(pd.DataFrame({"grid_id": grade["grid_id"], "composto_inicio": inicios[0],
                                    "composto_fim": fins[0], "disponivel_utc": disponivel,
                                    "ndvi": ndvi, "evi": evi, "frac_qualidade": frac, "pixels_validos": n}))
        if numero % 5 == 0 or numero == len(por_inicio):
            print(f"Compostos de vegetacao: {numero}/{len(por_inicio)}", flush=True)
    if not quadros:
        raise ValueError("Nenhum composto de vegetacao disponivel.")
    return pd.concat(quadros, ignore_index=True)


def processar(pasta=PASTA, grade_path=GRADE, sufixo=""):
    pasta = Path(pasta)
    grade = pd.read_csv(grade_path).sort_values(["linha", "coluna"]).reset_index(drop=True)
    if len(grade) != 256 or grade["grid_id"].duplicated().any():
        raise ValueError("Grade geografica invalida.")
    manifesto, origem = ler_manifesto(pasta)
    altitudes = relevo(manifesto, origem, grade)
    vegetacoes = vegetacao(manifesto, origem, grade)
    if sufixo and not re.fullmatch(r"_[a-zA-Z0-9_]+", sufixo):
        raise ValueError("Sufixo de saida invalido.")
    relevo_path, veg_path = pasta / f"relevo_grade{sufixo}.parquet", pasta / f"vegetacao_grade{sufixo}.parquet"
    altitudes.to_parquet(relevo_path, index=False)
    vegetacoes.to_parquet(veg_path, index=False)
    resumo = {"celulas_relevo": len(altitudes), "celulas_com_relevo": int(altitudes["altitude_media_m"].notna().sum()),
              "linhas_vegetacao": len(vegetacoes), "compostos": int(vegetacoes["composto_inicio"].nunique()),
              "valores_ndvi_validos": int(vegetacoes["ndvi"].notna().sum()),
              "sha256_relevo": sha256(relevo_path), "sha256_vegetacao": sha256(veg_path),
              "criterio_qualidade": "MODLAND=0, confiabilidade 0-2, terra, sem nuvem adjacente/mista, neve ou sombra; minimo 20% dos pixels"}
    (pasta / f"resumo_geografia{sufixo}.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    return resumo


def main():
    parser = argparse.ArgumentParser(description="Agrega vegetacao e relevo NASA por celula.")
    parser.add_argument("--pasta", type=Path, default=PASTA)
    parser.add_argument("--grade", type=Path, default=GRADE)
    parser.add_argument("--sufixo", default="")
    args = parser.parse_args()
    print(json.dumps(processar(args.pasta, args.grade, args.sufixo), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
