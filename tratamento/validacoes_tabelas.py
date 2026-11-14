from pathlib import Path
import argparse
import csv
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
EVENTOS = "acq_timestamp_utc sensor_family satellite proc_level latitude longitude country state daynight confidence_norm pixel_area_km2 bt_mir_kelvin bt_tir_kelvin thermal_anomaly_delta_k normalized_thermal_index frp_mw frp_density_mw_km2 confidence_weighted_frp".split()
CELULAS = "grid_id acq_date grid_lat grid_lon country state hotspot_count night_event_ratio total_frp_mw mean_frp_mw max_frp_mw mean_frp_density_mw_km2 max_frp_density_mw_km2 sum_confidence_weighted_frp max_thermal_anomaly_delta_k mean_normalized_thermal_index sp_coverage_ratio frp_density_daily_growth_rate rolling_3d_frp_max rolling_3d_hotspot_sum temp_2m_c temp_2m_max_c rel_humidity_2m_pct precip_mm_day wind_speed_10m_ms soil_moisture_top soil_moisture_root modis_valid_obs_ratio viirs_valid_obs_ratio has_valid_satellite_obs ndvi evi land_cover_class tree_cover_pct elevation_m".split()
TEXTOS = set("acq_timestamp_utc acq_date sensor_family satellite proc_level country state daynight grid_id has_valid_satellite_obs".split())
LIMITES = {"latitude": (-90, 90), "longitude": (-180, 180), "grid_lat": (-90, 90), "grid_lon": (-180, 180), "confidence_norm": (0, 1), "night_event_ratio": (0, 1), "sp_coverage_ratio": (0, 1), "modis_valid_obs_ratio": (0, 1), "viirs_valid_obs_ratio": (0, 1), "soil_moisture_top": (0, 1), "soil_moisture_root": (0, 1), "rel_humidity_2m_pct": (0, 100), "tree_cover_pct": (0, 100), "ndvi": (-1, 1), "evi": (-1, 1), "normalized_thermal_index": (-1, 1)}
NAO_NEGATIVAS = set("hotspot_count rolling_3d_hotspot_sum pixel_area_km2 bt_mir_kelvin bt_tir_kelvin frp_mw frp_density_mw_km2 confidence_weighted_frp total_frp_mw mean_frp_mw max_frp_mw mean_frp_density_mw_km2 max_frp_density_mw_km2 sum_confidence_weighted_frp rolling_3d_frp_max precip_mm_day wind_speed_10m_ms".split())


def ler_csv(caminho, colunas):
    caminho = Path(caminho)
    if not caminho.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    with caminho.open(encoding="utf-8-sig", newline="") as arquivo:
        cabecalho = next(csv.reader(arquivo), [])
    repetidas = sorted({c for c in cabecalho if cabecalho.count(c) > 1})
    ausentes = sorted(set(colunas) - set(cabecalho))
    if repetidas or ausentes:
        raise ValueError(f"{caminho.name}: cabeçalhos duplicados: {repetidas}; colunas ausentes: {ausentes}")
    dados = pd.read_csv(caminho, dtype="string", encoding="utf-8-sig", keep_default_na=False)
    if dados.empty:
        raise ValueError(f"A tabela {caminho.name} não contém linhas.")
    return dados


def validar_tabela(dados, tipo):
    if tipo not in {"eventos", "celulas"}:
        raise ValueError("Tipo deve ser eventos ou celulas.")
    esperadas = EVENTOS if tipo == "eventos" else CELULAS
    faltantes = sorted(set(esperadas) - set(dados.columns))
    if faltantes:
        raise ValueError(f"Colunas ausentes: {faltantes}")
    df = dados.copy()
    problemas = []

    def registrar(regra, mascara, bloqueante=True):
        quantidade = int(mascara.fillna(False).sum())
        if quantidade:
            problemas.append({"regra": regra, "linhas": quantidade, "bloqueante": bloqueante})

    for coluna in df.select_dtypes(include=["object", "string"]).columns:
        df[coluna] = df[coluna].astype("string").str.strip().replace("", pd.NA)
    for coluna in sorted(set(esperadas) - TEXTOS):
        valor = pd.to_numeric(df[coluna], errors="coerce")
        registrar(f"{coluna}: valor não numérico", df[coluna].notna() & valor.isna())
        registrar(f"{coluna}: valor não finito", valor.notna() & ~np.isfinite(valor))
        df[coluna] = valor
        if coluna in LIMITES:
            minimo, maximo = LIMITES[coluna]
            registrar(f"{coluna}: fora de [{minimo}, {maximo}]", valor.notna() & ~valor.between(minimo, maximo))
        if coluna in NAO_NEGATIVAS:
            registrar(f"{coluna}: valor negativo", valor < 0)
        if coluna in {"hotspot_count", "rolling_3d_hotspot_sum", "land_cover_class"}:
            registrar(f"{coluna}: valor não inteiro", valor.notna() & (valor % 1 != 0))
    data = "acq_timestamp_utc" if tipo == "eventos" else "acq_date"
    convertido = pd.to_datetime(df[data], errors="coerce", utc=True)
    registrar(f"{data}: data ausente ou inválida", convertido.isna())
    if tipo == "celulas":
        registrar("acq_date: horário diferente de meia-noite UTC", convertido.notna() & (convertido != convertido.dt.normalize()))
    df[data] = convertido
    obrigatorias = ["latitude", "longitude", "sensor_family", "satellite"] if tipo == "eventos" else ["grid_id", "grid_lat", "grid_lon", "hotspot_count"]
    for coluna in obrigatorias:
        registrar(f"{coluna}: valor obrigatório ausente", df[coluna].isna())
    if tipo == "eventos":
        for coluna, valores in {"sensor_family": {"MODIS", "VIIRS"}, "proc_level": {"SP", "NRT"}, "daynight": {"D", "N"}}.items():
            df[coluna] = df[coluna].str.upper()
            registrar(f"{coluna}: categoria desconhecida", df[coluna].notna() & ~df[coluna].isin(valores))
        registrar("eventos duplicados exatos", df.duplicated(keep=False), False)
    else:
        booleanos = df["has_valid_satellite_obs"].astype("string").str.lower().map({"true": True, "false": False, "1": True, "0": False})
        registrar("has_valid_satellite_obs: booleano inválido", df["has_valid_satellite_obs"].notna() & booleanos.isna())
        df["has_valid_satellite_obs"] = booleanos.astype("boolean")
        registrar("grid_id e acq_date duplicados", df.duplicated(["grid_id", "acq_date"], keep=False))
        registrar("focos sem observação válida", df["hotspot_count"].gt(0) & df["has_valid_satellite_obs"].eq(False), False)
    return df, pd.DataFrame(problemas, columns=["regra", "linhas", "bloqueante"])


def carregar_e_validar(caminho_eventos, caminho_celulas):
    eventos, rel_eventos = validar_tabela(ler_csv(caminho_eventos, EVENTOS), "eventos")
    celulas, rel_celulas = validar_tabela(ler_csv(caminho_celulas, CELULAS), "celulas")
    return eventos, celulas, rel_eventos, rel_celulas


def exibir_relatorio(nome, dados, problemas):
    data = "acq_timestamp_utc" if nome == "eventos" else "acq_date"
    print(f"\n{nome}: {len(dados):,} linhas")
    print(f"Período: {dados[data].min()} a {dados[data].max()}")
    ausentes = dados.isna().sum()
    print("Valores ausentes:")
    print(ausentes[ausentes > 0].sort_values(ascending=False).to_string())
    print("Inconsistências:")
    print(problemas.to_string(index=False) if not problemas.empty else "Nenhuma inconsistência encontrada.")


def argumentos():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eventos", type=Path, default=RAIZ / "dados" / "dados_previsao.csv")
    parser.add_argument("--celulas", type=Path, default=RAIZ / "dados" / "dados_satelites.csv")
    return parser


def main():
    args = argumentos().parse_args()
    eventos, celulas, rel_eventos, rel_celulas = carregar_e_validar(args.eventos, args.celulas)
    exibir_relatorio("eventos", eventos, rel_eventos)
    exibir_relatorio("celulas", celulas, rel_celulas)
    print("\nDistribuição de focos:")
    print(celulas["hotspot_count"].describe().to_string())
    print("\nObservação válida:")
    print(celulas["has_valid_satellite_obs"].value_counts(dropna=False).to_string())
    if any(not r.empty and r["bloqueante"].any() for r in (rel_eventos, rel_celulas)):
        raise SystemExit("Validação reprovada: corrija as inconsistências bloqueantes.")


if __name__ == "__main__":
    main()
