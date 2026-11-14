import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from modelo.config import ALVO
from modelo.dados import montar_base
from modelo.prever_viirs import prever
from modelo.prever_atual import UNIDADES, extrair_clima, prever_atual
from modelo.treinar_viirs import particionar, treinar
from modelo.viirs_dados import BASE, ler, validar


def test_base_generica_preserva_positivos_e_exclui_zero_desconhecido():
    dados = pd.DataFrame([
        ("a", "2025-01-01", 1, None), ("a", "2025-01-02", 2, None),
        ("b", "2025-01-01", 1, None), ("b", "2025-01-02", 0, True),
        ("c", "2025-01-01", 1, None), ("c", "2025-01-02", 0, None),
        ("d", "2025-01-01", 0, None), ("d", "2025-01-02", 1, None),
    ], columns=["grid_id", "acq_date", "hotspot_count", "has_valid_satellite_obs"])
    dados["acq_date"] = pd.to_datetime(dados["acq_date"], utc=True)
    dados["has_valid_satellite_obs"] = dados["has_valid_satellite_obs"].astype("boolean")
    resultado = montar_base(dados).set_index("grid_id")[ALVO].to_dict()
    assert resultado == {"a": 1, "b": 0}


def test_mascaras_incompletas_nao_viram_negativos():
    dados = ler(BASE)
    incompletas = dados[~dados["fully_observed"]]
    assert incompletas["event"].isna().all()
    assert len(incompletas[incompletas["fire_pixels"] > 0]) > 0
    assert dados.loc[dados["fully_observed"], "event"].isin([0, 1]).all()


@pytest.mark.parametrize("tipo", ["clima_futuro", "data_granulo", "contagem_classe"])
def test_validacao_rejeita_inconsistencias_e_vazamento(tipo):
    dados = ler(BASE).head(1).copy()
    if tipo == "clima_futuro":
        dados.loc[dados.index[0], "weather_T2M"] = 28.0
        dados.loc[dados.index[0], "weather_last_date"] = dados.iloc[0]["period_start"]
    elif tipo == "data_granulo":
        dados.loc[dados.index[0], "period_start"] += pd.Timedelta(days=1)
        dados.loc[dados.index[0], "period_end_exclusive"] += pd.Timedelta(days=1)
    else:
        dados.loc[dados.index[0], "class_3"] += 1
    with pytest.raises(ValueError):
        validar(dados)


def test_divisao_temporal_e_espacial():
    partes = particionar(ler(BASE))
    assert partes["treino"]["period_start"].max() < partes["calibracao"]["period_start"].min()
    assert partes["calibracao"]["period_start"].max() < partes["decisao"]["period_start"].min()
    assert partes["decisao"]["period_start"].max() < partes["teste_local"]["period_start"].min()
    assert set(partes["treino"]["region_id"]).isdisjoint(partes["teste_espacial"]["region_id"])
    assert all(parte["fully_observed"].all() for parte in partes.values())


def test_treino_e_consulta_historica_reproduzem_mesma_probabilidade(tmp_path, monkeypatch):
    relatorio = treinar(BASE, tmp_path)
    assert relatorio["metricas_teste"]["geral"]["exemplos"] > 0
    tabela = pd.read_csv(tmp_path / "previsoes_teste_viirs.csv")
    linha = tabela.iloc[0]
    data = (pd.Timestamp(linha["period_start"]) - pd.Timedelta(days=1)).date().isoformat()
    resposta = prever(BASE, tmp_path / "modelo_viirs.joblib", "Brasil", data, linha["region_name"])
    assert resposta["situacao"] == "experimental_retrospectivo"
    assert resposta["probabilidade_incendio_incondicional"] is None
    assert resposta["recortes"][0]["probabilidade_condicional"] == pytest.approx(linha["probabilidade_condicional"])
    with pytest.raises(ValueError, match="Brasil"):
        prever(BASE, tmp_path / "modelo_viirs.joblib", "Japao", data)
    with pytest.raises(ValueError, match="[Nn]ao ha recortes"):
        prever(BASE, tmp_path / "modelo_viirs.joblib", "Brasil", "2026-11-13")
    arquivo = tmp_path / "relatorio_viirs.json"
    assert json.loads(arquivo.read_text(encoding="utf-8"))["sha256_dados"] == relatorio["sha256_dados"]
    clima = {"weather_T2M": 26.0, "weather_RH2M": 60.0, "weather_PRECTOTCORR": 20.0,
             "weather_WS10M": 2.0, "weather_GWETROOT": 0.5}
    monkeypatch.setattr("modelo.prever_atual.consultar_clima", lambda *args: (clima, "a" * 64))
    atual = prever_atual(BASE, tmp_path / "modelo_viirs.joblib", "Brasil", "centro_norte")
    assert atual["situacao"] == "experimental_nao_operacional"
    assert atual["probabilidade_incendio_incondicional"] is None
    assert 0 <= atual["recortes"][0]["probabilidade_condicional"] <= 1
    def indisponivel(*args):
        raise RuntimeError("Fonte indisponivel")
    monkeypatch.setattr("modelo.prever_atual.consultar_clima", indisponivel)
    atual = prever_atual(BASE, tmp_path / "modelo_viirs.joblib", "Brasil", "centro_norte")
    assert atual["recortes"][0]["situacao"] == "dados_indisponiveis"
    assert atual["recortes"][0]["probabilidade_condicional"] is None


def test_clima_exige_oito_dias_e_unidades_corretas():
    inicio, fim = date(2025, 1, 1), date(2025, 1, 8)
    dias = pd.date_range(inicio, fim).strftime("%Y%m%d")
    corpo = {
        "header": {"time_standard": "UTC", "start": "20250101", "end": "20250108", "fill_value": -999},
        "parameters": {nome: {"units": unidade} for nome, unidade in UNIDADES.items()},
        "properties": {"parameter": {nome: {dia: 1.0 for dia in dias} for nome in UNIDADES}},
    }
    assert extrair_clima(corpo, inicio, fim)["weather_PRECTOTCORR"] == 8.0
    corpo["properties"]["parameter"]["T2M"]["20250108"] = -999
    with pytest.raises(ValueError, match="incompleta"):
        extrair_clima(corpo, inicio, fim)


def test_consulta_rejeita_base_alterada(tmp_path):
    pasta = Path("modelo/artefatos/viirs_regional")
    if not (pasta / "modelo_viirs.joblib").exists():
        pytest.skip("Artefato regional ainda nao treinado.")
    copia = tmp_path / "alterada.csv"
    copia.write_bytes(BASE.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="mesma base auditada"):
        prever(copia, pasta / "modelo_viirs.joblib", "Brasil", "2025-10-21")
