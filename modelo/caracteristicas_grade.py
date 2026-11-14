import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from modelo.coletar_clima import UNIDADES, celulas_meteorologicas
from modelo.coletar_mascaras import PASTA, sha256
from modelo.grade import SAIDA as GRADE

CLIMA = ["T2M", "RH2M", "PRECTOTCORR", "WS10M", "GWETROOT"]
BASE = ["latitude", "longitude", "dia_seno", "dia_cosseno"]
HISTORICO_MASCARA = ["mascara_foco_2d", "mascara_foco_3d", "mascara_focos_7d", "mascara_conhecidos_7d", "mascara_focos_14d", "mascara_conhecidos_14d"]
HISTORICO_FIRMS = ["viirs_registros_7d", "viirs_frp_7d", "modis_registros_7d", "modis_frp_7d"]
VEGETACAO = ["vegetacao_ndvi", "vegetacao_evi", "vegetacao_fracao_qualidade", "vegetacao_idade_dias"]
RELEVO = ["altitude_media_m", "altitude_desvio_m"]
CARACTERISTICAS = BASE + ["clima_" + nome for nome in CLIMA] + HISTORICO_MASCARA + HISTORICO_FIRMS + VEGETACAO + RELEVO


def clima_janelas(clima):
    dados = clima.copy()
    dados["data_utc"] = pd.to_datetime(dados["data_utc"], utc=True, errors="raise")
    dados = dados.sort_values(["lat_power", "lon_power", "data_utc"]).reset_index(drop=True)
    if dados.duplicated(["lat_power", "lon_power", "data_utc"]).any():
        raise ValueError("Clima duplicado para celula e dia.")
    partes = []
    for _, grupo in dados.groupby(["lat_power", "lon_power"], sort=False):
        if not grupo["data_utc"].diff().dropna().eq(pd.Timedelta(days=1)).all():
            raise ValueError("A serie climatica precisa preservar dias ausentes.")
        saida = grupo[["lat_power", "lon_power", "data_utc"]].copy()
        for nome in CLIMA:
            v = pd.to_numeric(grupo[nome], errors="raise")
            rolagem = v.rolling(8, min_periods=8)
            saida["clima_" + nome] = (rolagem.sum() if nome == "PRECTOTCORR" else rolagem.mean()).shift(3)
        partes.append(saida)
    return pd.concat(partes, ignore_index=True)


def adicionar_historico_mascaras(dados):
    saida = dados.sort_values(["grid_id", "data_utc"]).copy()
    grupo = saida.groupby("grid_id", sort=False)
    emissao = saida["data_utc"] - pd.Timedelta(hours=1)
    estados = {}
    for atraso in range(2, 16):
        data_anterior = grupo["data_utc"].shift(atraso)
        producao = grupo["producao_utc"].shift(atraso)
        fogo = grupo["evidencia_foco"].shift(atraso).astype("boolean").fillna(False).astype(bool)
        completa = grupo["mascara_completa"].shift(atraso).astype("boolean").fillna(False).astype(bool)
        anterior = data_anterior.eq(saida["data_utc"] - pd.Timedelta(days=atraso))
        disponivel = anterior & producao.notna() & producao.le(emissao)
        conhecido = disponivel & (fogo | completa)
        estados[atraso] = pd.Series(np.where(conhecido, fogo.astype(float), np.nan), index=saida.index)
    saida["mascara_foco_2d"] = estados[2]
    saida["mascara_foco_3d"] = estados[3]
    for janela in [7, 14]:
        bloco = pd.concat([estados[i] for i in range(2, janela + 2)], axis=1)
        saida[f"mascara_focos_{janela}d"] = bloco.sum(axis=1, min_count=1)
        saida[f"mascara_conhecidos_{janela}d"] = bloco.notna().sum(axis=1)
    return saida


def agregar_firms(firms):
    dados = firms.copy()
    dados["instante_utc"] = pd.to_datetime(dados["instante_utc"], utc=True, errors="raise")
    dados["linha"] = np.floor((dados["latitude"] + 12) / 0.25).astype(int)
    dados["coluna"] = np.floor((dados["longitude"] + 47) / 0.25).astype(int)
    dados = dados[dados["linha"].between(0, 15) & dados["coluna"].between(0, 15)].copy()
    dados["grid_id"] = [f"matopiba_{i:02d}_{j:02d}" for i, j in zip(dados["linha"], dados["coluna"])]
    dados["data_utc"] = dados["instante_utc"].dt.normalize()
    dados["tipo"] = np.where(dados["fonte"].str.startswith("VIIRS_NOAA20"), "viirs", "modis")
    grupo = dados.groupby(["grid_id", "data_utc", "tipo"], sort=False)
    contagem = grupo.size().rename("registros")
    frp = grupo["frp"].sum(min_count=1).rename("frp")
    return pd.concat([contagem, frp], axis=1).reset_index()


def adicionar_firms(dados, firms):
    saida = dados.sort_values(["grid_id", "data_utc"]).copy()
    agregados = agregar_firms(firms)
    for tipo in ["viirs", "modis"]:
        parte = agregados[agregados["tipo"].eq(tipo)].drop(columns="tipo").rename(columns={"registros": "_registros", "frp": "_frp"})
        saida = saida.merge(parte, on=["grid_id", "data_utc"], how="left", validate="one_to_one", sort=False)
        grupo = saida.groupby("grid_id", sort=False)
        registros = pd.concat([grupo["_registros"].shift(i) for i in range(2, 9)], axis=1)
        frp = pd.concat([grupo["_frp"].shift(i) for i in range(2, 9)], axis=1)
        saida[f"{tipo}_registros_7d"] = registros.sum(axis=1, min_count=1)
        saida[f"{tipo}_frp_7d"] = frp.sum(axis=1, min_count=1)
        saida = saida.drop(columns=["_registros", "_frp"])
    return saida


def adicionar_vegetacao(dados, vegetacao):
    fontes = vegetacao.copy()
    fontes["disponivel_utc"] = pd.to_datetime(fontes["disponivel_utc"], utc=True, errors="raise").astype("datetime64[ns, UTC]")
    fontes["composto_fim"] = pd.to_datetime(fontes["composto_fim"], utc=True, errors="raise").astype("datetime64[ns, UTC]")
    if fontes.duplicated(["grid_id", "disponivel_utc"]).any():
        raise ValueError("Composto de vegetacao duplicado para a mesma celula e disponibilidade.")
    if (fontes["disponivel_utc"] < fontes["composto_fim"] + pd.Timedelta(days=1)).any():
        raise ValueError("Vegetacao marcada como disponivel antes de terminar o composto.")
    fontes = fontes[fontes["ndvi"].notna() & fontes["evi"].notna()].copy()
    fontes = fontes[["grid_id", "disponivel_utc", "composto_fim", "ndvi", "evi", "frac_qualidade"]].rename(
        columns={"ndvi": "vegetacao_ndvi", "evi": "vegetacao_evi", "frac_qualidade": "vegetacao_fracao_qualidade"})
    entrada = dados.copy()
    entrada["emissao_utc"] = entrada["data_utc"] - pd.Timedelta(hours=1)
    saida = pd.merge_asof(entrada.sort_values("emissao_utc"), fontes.sort_values("disponivel_utc"),
                          left_on="emissao_utc", right_on="disponivel_utc", by="grid_id",
                          direction="backward", tolerance=pd.Timedelta(days=60))
    saida["vegetacao_idade_dias"] = (saida["emissao_utc"] - (saida["composto_fim"] + pd.Timedelta(days=1))).dt.total_seconds() / 86400
    return saida.drop(columns=["emissao_utc"])


def montar(mascaras, clima, grade, firms=None, vegetacao=None, relevo=None):
    dados = mascaras.copy()
    dados["data_utc"] = pd.to_datetime(dados["data_utc"], utc=True, errors="raise")
    dados["producao_utc"] = pd.to_datetime(dados["producao_utc"], utc=True, errors="coerce")
    if dados.duplicated(["grid_id", "data_utc"]).any():
        raise ValueError("Mascara duplicada por celula e data.")
    for _, grupo in dados.sort_values(["grid_id", "data_utc"]).groupby("grid_id", sort=False):
        if not grupo["data_utc"].diff().dropna().eq(pd.Timedelta(days=1)).all():
            raise ValueError("A base de mascaras deve conter todos os dias, inclusive desconhecidos.")
    dados = adicionar_historico_mascaras(dados)
    if firms is not None:
        dados = adicionar_firms(dados, firms)
    else:
        for coluna in HISTORICO_FIRMS:
            dados[coluna] = np.nan
    mapa = celulas_meteorologicas(grade)[["grid_id", "lat_power", "lon_power"]]
    dados = dados.merge(mapa, on="grid_id", how="left", validate="many_to_one")
    janelas = clima_janelas(clima)
    dados = dados.merge(janelas, on=["lat_power", "lon_power", "data_utc"], how="left", validate="many_to_one")
    if vegetacao is not None:
        dados = adicionar_vegetacao(dados, vegetacao)
    else:
        for coluna in VEGETACAO:
            dados[coluna] = np.nan
    if relevo is not None:
        if relevo["grid_id"].duplicated().any():
            raise ValueError("Relevo duplicado para uma celula.")
        dados = dados.merge(relevo[["grid_id", *RELEVO]], on="grid_id", how="left", validate="many_to_one")
    else:
        for coluna in RELEVO:
            dados[coluna] = np.nan
    dia = dados["data_utc"].dt.dayofyear
    duracao = np.where(dados["data_utc"].dt.is_leap_year, 366.0, 365.0)
    dados["dia_seno"] = np.sin(2 * np.pi * (dia - 1) / duracao)
    dados["dia_cosseno"] = np.cos(2 * np.pi * (dia - 1) / duracao)
    return dados.sort_values(["data_utc", "grid_id"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Prepara variaveis historicas para a grade piloto.")
    parser.add_argument("--mascaras", type=Path, required=True)
    parser.add_argument("--clima", type=Path, required=True)
    parser.add_argument("--firms", type=Path)
    parser.add_argument("--vegetacao", type=Path)
    parser.add_argument("--relevo", type=Path)
    parser.add_argument("--grade", type=Path, default=GRADE)
    parser.add_argument("--saida", type=Path, default=PASTA / "caracteristicas_grade.parquet")
    args = parser.parse_args()
    firmas = pd.read_parquet(args.firms) if args.firms else None
    vegetacao = pd.read_parquet(args.vegetacao) if args.vegetacao else None
    relevo = pd.read_parquet(args.relevo) if args.relevo else None
    dados = montar(pd.read_parquet(args.mascaras), pd.read_parquet(args.clima), pd.read_csv(args.grade), firmas, vegetacao, relevo)
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    dados.to_parquet(args.saida, index=False)
    resumo = {"linhas": len(dados), "rotulos_conhecidos": int(dados["alvo_condicional"].notna().sum()),
              "positivos": int(dados["alvo_condicional"].fillna(0).sum()),
              "fontes": {"mascaras": sha256(args.mascaras), "clima": sha256(args.clima),
                         "grade": sha256(args.grade), "firms": sha256(args.firms) if args.firms else None,
                         "vegetacao": sha256(args.vegetacao) if args.vegetacao else None,
                         "relevo": sha256(args.relevo) if args.relevo else None},
              "sha256_saida": sha256(args.saida),
              "limite": "Dados historicos podem ter sido revisados; SP nao comprova disponibilidade NRT no instante historico."}
    args.saida.with_suffix(".json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(resumo, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
