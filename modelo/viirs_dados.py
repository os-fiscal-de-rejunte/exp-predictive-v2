import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from modelo.config import RAIZ

ORIGEM = RAIZ.parent / "HACKATON-NASA" / "runs" / "viirs_daily_v4b_20261111" / "processed" / "mask_examples.parquet"
BASE = RAIZ / "dados" / "base_viirs_regional.csv"
METADADOS = RAIZ / "dados" / "base_viirs_regional_metadados.json"
ARTEFATOS = RAIZ / "modelo" / "artefatos" / "viirs_regional"

CARACTERISTICAS = [
    "latitude", "longitude", "dia_seno", "dia_cosseno",
    "past_viirs_mask_event", "past_viirs_log_records", "past_modis_event",
    "weather_T2M", "weather_RH2M", "weather_PRECTOTCORR",
    "weather_WS10M", "weather_GWETROOT",
]
DATAS = [
    "period_start", "period_end_exclusive", "weather_last_date", "viirs_history_end",
    "past_viirs_mask_date", "past_viirs_mask_produced_at",
    "past_modis_period_end", "past_modis_produced_at",
]
CLASSES = [f"class_{i}" for i in range(10)]
COLUNAS = [
    "region_name", "region_id", "latitude", "longitude", "spatial_group",
    "xllcorner", "yllcorner", "ncols", "nrows", "cellsize_m",
    *DATAS[:2], *CLASSES, "masked_pixels", "invalid_pixels", "uncovered_pixels",
    "total_pixels", "valid_pixels", "fire_pixels", "fully_observed", "event",
    "production_time", "source_file", "source_sha256", "source_files", "source_hashes",
    "source_tiles_complete", "past_viirs_mask_event", "past_viirs_mask_date",
    "past_viirs_mask_produced_at", "past_viirs_records", "past_viirs_log_records",
    "viirs_history_start", "viirs_history_end", "past_modis_event",
    "past_modis_period_end", "past_modis_produced_at", "weather_last_date",
    "weather_T2M", "weather_RH2M", "weather_PRECTOTCORR", "weather_WS10M",
    "weather_GWETROOT",
]


def sha256(caminho):
    resumo = hashlib.sha256()
    with Path(caminho).open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            resumo.update(bloco)
    return resumo.hexdigest()


def _booleano(valores, nome):
    convertido = valores.astype("string").str.lower().map({"true": True, "false": False, "1": True, "0": False})
    if convertido.isna().any():
        raise ValueError(f"Valores invalidos ou ausentes em {nome}.")
    return convertido.astype(bool)


def validar(dados):
    faltam = sorted(set(COLUNAS) - set(dados.columns))
    if faltam:
        raise ValueError(f"Colunas ausentes: {faltam}")
    dados = dados.copy()
    for coluna in DATAS:
        dados[coluna] = pd.to_datetime(dados[coluna], utc=True, errors="raise")
    for coluna in ["fully_observed", "source_tiles_complete"]:
        dados[coluna] = _booleano(dados[coluna], coluna)
    numericas = list(set(CLASSES + [
        "latitude", "longitude", "ncols", "nrows", "cellsize_m", "masked_pixels",
        "invalid_pixels", "uncovered_pixels", "total_pixels", "valid_pixels", "fire_pixels",
        "event", "past_viirs_mask_event", "past_viirs_records", "past_viirs_log_records",
        "past_modis_event", "weather_T2M", "weather_RH2M", "weather_PRECTOTCORR",
        "weather_WS10M", "weather_GWETROOT",
    ]))
    for coluna in numericas:
        dados[coluna] = pd.to_numeric(dados[coluna], errors="raise")
        if np.isinf(dados[coluna].to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError(f"Valor infinito em {coluna}.")
    if dados.empty or dados[["region_id", "region_name", "period_start"]].isna().any().any():
        raise ValueError("Regiao e data sao obrigatorias.")
    if dados.duplicated(["region_id", "period_start"]).any():
        raise ValueError("Existe regiao e dia duplicado.")
    if not dados["period_start"].dt.normalize().eq(dados["period_start"]).all():
        raise ValueError("As datas de alvo devem ser dias UTC inteiros.")
    if not dados["period_end_exclusive"].eq(dados["period_start"] + pd.Timedelta(days=1)).all():
        raise ValueError("O intervalo do produto deve corresponder a um dia UTC.")
    if not dados["latitude"].between(-90, 90).all() or not dados["longitude"].between(-180, 180).all():
        raise ValueError("Coordenadas invalidas.")
    if not dados["spatial_group"].isin(["train", "holdout"]).all():
        raise ValueError("Grupo espacial desconhecido.")
    for coluna in CLASSES + ["masked_pixels", "total_pixels", "valid_pixels", "fire_pixels"]:
        v = dados[coluna]
        if v.isna().any() or (v < 0).any() or (v % 1 != 0).any():
            raise ValueError(f"Contagem invalida em {coluna}.")
    if (dados["total_pixels"] <= 0).any() or not dados["total_pixels"].eq(dados["ncols"] * dados["nrows"]).all() or (dados["cellsize_m"] <= 0).any():
        raise ValueError("Geometria ou tamanho da mascara invalido.")
    if not dados[CLASSES].sum(axis=1).add(dados["masked_pixels"]).eq(dados["total_pixels"]).all():
        raise ValueError("Contagens da mascara nao somam o total de pixels.")
    validos = dados[["class_3", "class_5", "class_7", "class_8", "class_9"]].sum(axis=1)
    focos = dados[["class_7", "class_8", "class_9"]].sum(axis=1)
    if not validos.eq(dados["valid_pixels"]).all() or not focos.eq(dados["fire_pixels"]).all():
        raise ValueError("Contagens de classes validas ou de focos inconsistentes.")
    completos = validos.eq(dados["total_pixels"])
    if not completos.eq(dados["fully_observed"]).all():
        raise ValueError("Indicador de mascara completa inconsistente.")
    esperado = focos.gt(0).astype(int)
    if dados.loc[~completos, "event"].notna().any() or not dados.loc[completos, "event"].eq(esperado[completos]).all():
        raise ValueError("Rotulos inconsistentes com a mascara integralmente observada.")
    nomes = dados["source_file"].astype(str)
    if not nomes.str.fullmatch(r"VJ114A1\.A\d{7}\.h\d{2}v\d{2}\.002\.\d{13}\.h5").all():
        raise ValueError("O produto de entrada nao e VJ114A1.")
    datas_granulos = pd.to_datetime(nomes.str.slice(9, 16), format="%Y%j", utc=True, errors="raise")
    if not datas_granulos.eq(dados["period_start"]).all():
        raise ValueError("A data do granulo difere da data de observacao.")
    if not dados["source_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("Hash do granulo ausente ou invalido.")
    d = dados["period_start"]
    clima = dados[["weather_T2M", "weather_RH2M", "weather_PRECTOTCORR", "weather_WS10M", "weather_GWETROOT"]].notna().any(axis=1)
    if (clima & (dados["weather_last_date"].isna() | (dados["weather_last_date"] > d - pd.Timedelta(days=3)))).any():
        raise ValueError("Clima sem data comprovada ate D-3.")
    historico = dados["past_viirs_log_records"].notna()
    if (historico & (dados["viirs_history_end"].isna() | (dados["viirs_history_end"] > d - pd.Timedelta(days=2)))).any():
        raise ValueError("Historico VIIRS sem data comprovada ate D-2.")
    contagem = dados["past_viirs_records"]
    if (contagem.notna() != historico).any() or (contagem.dropna() <= 0).any():
        raise ValueError("O historico de deteccoes ausente deve permanecer desconhecido.")
    if not np.allclose(np.log1p(contagem[historico]), dados.loc[historico, "past_viirs_log_records"]):
        raise ValueError("Contagem e logaritmo do historico VIIRS inconsistentes.")
    modis = dados["past_modis_event"].notna()
    if (modis & (dados["past_modis_period_end"].isna() | dados["past_modis_produced_at"].isna() | (dados["past_modis_period_end"] > d) | (dados["past_modis_produced_at"] >= d))).any():
        raise ValueError("Mascara MODIS anterior indisponivel no instante previsto.")
    viirs = dados["past_viirs_mask_event"].notna()
    if (viirs & (dados["past_viirs_mask_date"].isna() | dados["past_viirs_mask_produced_at"].isna() | (dados["past_viirs_mask_date"] >= d) | (dados["past_viirs_mask_produced_at"] >= d))).any():
        raise ValueError("Mascara VIIRS anterior indisponivel no instante previsto.")
    for coluna in ["past_modis_event", "past_viirs_mask_event"]:
        if not dados[coluna].dropna().isin([0, 1]).all():
            raise ValueError(f"Historico binario invalido em {coluna}.")
    return dados.sort_values(["period_start", "region_id"]).reset_index(drop=True)


def ler(caminho):
    return validar(pd.read_csv(caminho))


def caracteristicas(dados):
    x = dados.copy()
    dia = x["period_start"].dt.dayofyear
    duracao = np.where(x["period_start"].dt.is_leap_year, 366.0, 365.0)
    angulo = 2 * np.pi * (dia - 1) / duracao
    x["dia_seno"] = np.sin(angulo)
    x["dia_cosseno"] = np.cos(angulo)
    return x[CARACTERISTICAS]


def preparar(origem, saida, metadados):
    origem, saida, metadados = map(Path, (origem, saida, metadados))
    resumo_path = origem.parent / "viirs_daily_summary.json"
    if len({origem.resolve(), saida.resolve(), metadados.resolve(), resumo_path.resolve()}) != 4:
        raise ValueError("Arquivos de origem, saida e metadados devem ser distintos.")
    resumo = json.loads(resumo_path.read_text(encoding="utf-8"))
    origem_sha = sha256(origem)
    if resumo.get("product") != "VJ114A1.002" or resumo.get("examples_sha256") != origem_sha:
        raise ValueError("Produto ou hash da origem nao corresponde ao manifesto anterior.")
    dados = pd.read_parquet(origem)
    dados = validar(dados)
    for coluna in ["source_files", "source_hashes"]:
        dados[coluna] = dados[coluna].map(lambda itens: "|".join(str(item) for item in itens))
    saida.parent.mkdir(parents=True, exist_ok=True)
    metadados.parent.mkdir(parents=True, exist_ok=True)
    dados[COLUNAS].to_csv(saida, index=False)
    conhecidos = dados[dados["fully_observed"]]
    descricao = {
        "produto": "VJ114A1.002 / VIIRS NOAA-20 / FireMask diario 1 km",
        "origem": str(origem.resolve()), "sha256_origem": origem_sha,
        "sha256_csv": sha256(saida), "linhas": len(dados),
        "mascaras_completas": len(conhecidos), "positivos_completos": int(conhecidos["event"].sum()),
        "negativos_completos": int(conhecidos["event"].eq(0).sum()),
        "mascaras_incompletas": int((~dados["fully_observed"]).sum()),
        "incompletas_com_foco": int(((~dados["fully_observed"]) & dados["fire_pixels"].gt(0)).sum()),
        "alvo": "Ao menos um pixel de foco no dia UTC seguinte, condicionado a todos os pixels da mascara terem classificacao valida.",
        "limites": "Oito areas selecionadas, amostra descontínua, arquivos retrospectivos possivelmente revisados; nao representa o Brasil.",
    }
    metadados.write_text(json.dumps(descricao, ensure_ascii=False, indent=2), encoding="utf-8")
    return descricao
