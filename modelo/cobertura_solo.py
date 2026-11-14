import argparse
import json
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
from pyhdf.SD import SD, SDC

from modelo.coletar_mascaras import PASTA, TILES, sha256
from modelo.grade import SAIDA as GRADE
from modelo.mascaras import mapa_pixels

COLUNAS = ["solo_floresta_frac", "solo_savana_campo_frac", "solo_cultivo_frac", "solo_qualidade_frac"]


def extrair(caminho, grade):
    arquivo = SD(os.path.relpath(caminho, Path.cwd()), SDC.READ)
    try:
        estrutura = arquivo.attributes()["StructMetadata.0"]
        if "Projection=GCTP_SNSOID" not in estrutura or "6371007.181" not in estrutura:
            raise ValueError("Projecao inesperada na cobertura do solo.")
        padrao = r"{}=\(([-0-9.]+),([-0-9.]+)\)"
        superior = re.search(padrao.format("UpperLeftPointMtrs"), estrutura)
        inferior = re.search(padrao.format("LowerRightMtrs"), estrutura)
        if not superior or not inferior:
            raise ValueError("Limites da grade MODIS ausentes.")
        x0, y0 = map(float, superior.groups())
        x1, y1 = map(float, inferior.groups())
        if not x0 < x1 or not y1 < y0:
            raise ValueError("Orientacao inesperada na cobertura do solo.")
        x = x0 + (np.arange(2400) + 0.5) * (x1 - x0) / 2400
        y = y0 + (np.arange(2400) + 0.5) * (y1 - y0) / 2400
        linhas, colunas, ids = mapa_pixels(x, y, grade)
        classes = arquivo.select("LC_Type1").get()[linhas, colunas]
        qualidade = arquivo.select("QC").get()[linhas, colunas]
        terra = arquivo.select("LW").get()[linhas, colunas]
        bom = (classes >= 1) & (classes <= 16) & (qualidade == 0) & (terra == 2)
        contagens = np.zeros((256, 17), dtype=np.int64)
        for classe in range(1, 17):
            contagens[:, classe] = np.bincount(ids[bom & (classes == classe)], minlength=256)
        return contagens, np.bincount(ids, minlength=256)
    finally:
        arquivo.end()


def processar(pasta=PASTA, grade_path=GRADE):
    pasta = Path(pasta)
    manifesto_path = pasta / "manifesto_cobertura_solo.json"
    manifesto = json.loads(manifesto_path.read_text(encoding="utf-8"))
    if manifesto.get("produto") != "MCD12Q1.061":
        raise ValueError("Produto de cobertura do solo inesperado.")
    grade = pd.read_csv(grade_path).sort_values(["linha", "coluna"]).reset_index(drop=True)
    if len(grade) != 256 or grade["grid_id"].duplicated().any():
        raise ValueError("Grade de cobertura do solo invalida.")
    por_ano = {}
    for item in manifesto["arquivos"].values():
        por_ano.setdefault(item["ano"], []).append(item)
    quadros = []
    for ano, itens in sorted(por_ano.items()):
        if len(itens) != len(TILES) or {x["tile"] for x in itens} != set(TILES):
            raise ValueError(f"Tiles incompletos para cobertura do solo em {ano}.")
        classes, esperados = np.zeros((256, 17), dtype=np.int64), np.zeros(256, dtype=np.int64)
        for item in itens:
            caminho = pasta / "brutos" / "cobertura_solo" / item["nome"]
            if sha256(caminho) != item["sha256"]:
                raise ValueError("Granulo de cobertura do solo foi alterado.")
            resultado, pixels = extrair(caminho, grade)
            classes += resultado
            esperados += pixels
        n = classes.sum(axis=1)
        qualidade = np.divide(n, esperados, out=np.zeros(256), where=esperados > 0)
        valido = (qualidade >= 0.2) & (n > 0)
        frac = lambda valores: np.divide(valores, n, out=np.full(256, np.nan), where=valido)
        dominante = np.where(valido, classes.argmax(axis=1), np.nan)
        disponibilidade = max([pd.Timestamp(f"{ano + 1}-01-01T00:00:00Z")] +
                              [pd.Timestamp(item[chave]) for item in itens for chave in ["producao_utc", "catalogo_atualizado_utc"]])
        quadros.append(pd.DataFrame({"grid_id": grade["grid_id"], "solo_ano": ano,
                                    "solo_disponivel_utc": disponibilidade, "solo_classe_dominante": dominante,
                                    "solo_floresta_frac": frac(classes[:, 1:6].sum(axis=1)),
                                    "solo_savana_campo_frac": frac(classes[:, 6:11].sum(axis=1)),
                                    "solo_cultivo_frac": frac(classes[:, 12] + classes[:, 14]),
                                    "solo_qualidade_frac": qualidade, "solo_pixels_validos": n}))
        print(f"Cobertura processada: {ano}", flush=True)
    if not quadros:
        raise ValueError("Nenhum ano de cobertura do solo disponivel.")
    dados = pd.concat(quadros, ignore_index=True)
    caminho = pasta / "cobertura_solo_grade.parquet"
    dados.to_parquet(caminho, index=False)
    resumo = {"produto": "MCD12Q1.061", "anos": sorted(por_ano), "celulas_ano": len(dados),
              "classificadas": int(dados["solo_classe_dominante"].notna().sum()),
              "criterio": "IGBP 1 a 16, QC=0 e LW=2; minimo 20% dos pixels da celula; mascara geografica nominal 500 m.",
              "disponibilidade": "Maior entre fim do ano, data de producao e atualizacao no catalogo dos dois tiles.",
              "sha256_grade": sha256(grade_path), "sha256_arquivo": sha256(caminho),
              "sha256_manifesto": sha256(manifesto_path)}
    (pasta / "resumo_cobertura_solo.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    return resumo


def adicionar(dados, cobertura):
    dados = dados.copy()
    solo = cobertura.copy()
    solo["solo_disponivel_utc"] = pd.to_datetime(solo["solo_disponivel_utc"], utc=True, errors="raise").astype("datetime64[ns, UTC]")
    if solo.duplicated(["grid_id", "solo_disponivel_utc"]).any():
        raise ValueError("Cobertura do solo duplicada na mesma data.")
    if (solo["solo_disponivel_utc"].dt.year <= solo["solo_ano"]).any():
        raise ValueError("Cobertura do solo disponivel antes do fim do ano.")
    dados["_emissao_utc"] = (pd.to_datetime(dados["data_utc"], utc=True) - pd.Timedelta(hours=1)).astype("datetime64[ns, UTC]")
    saida = pd.merge_asof(dados.sort_values("_emissao_utc"), solo.sort_values("solo_disponivel_utc"),
                          left_on="_emissao_utc", right_on="solo_disponivel_utc", by="grid_id", direction="backward")
    return saida.drop(columns="_emissao_utc").sort_values(["data_utc", "grid_id"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Agrega a cobertura anual MODIS na grade piloto.")
    parser.add_argument("--pasta", type=Path, default=PASTA)
    parser.add_argument("--grade", type=Path, default=GRADE)
    args = parser.parse_args()
    print(json.dumps(processar(args.pasta, args.grade), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
