from datetime import date

import numpy as np
import pandas as pd
import pytest

from modelo.caracteristicas_grade import adicionar_historico_mascaras, adicionar_vegetacao, clima_janelas
from modelo.coletar_firms import ler_csv, segmentos
from modelo.grade import criar_grade
from modelo.mascaras import RAIO, mapa_pixels
from modelo.treinar_grade import dividir


def test_grade_cobre_area_independentemente_dos_focos():
    grade = criar_grade()
    assert len(grade) == 256
    assert grade["grid_id"].is_unique
    assert grade["pais"].eq("Brazil").all()
    assert grade["fracao_brasil_aprox"].ge(0.999999).all()


def test_mapeamento_sinusoidal_atribui_celulas_corretas():
    grade = criar_grade()
    latitude = np.array([-11.875, -8.125])
    longitude = np.array([-46.875, -43.125])
    y = RAIO * np.deg2rad(latitude)
    x = RAIO * np.deg2rad(longitude) * np.cos(np.deg2rad(latitude))
    linhas, colunas, ids = mapa_pixels(x, y, grade)
    assert (0, 0, 0) in set(zip(linhas.tolist(), colunas.tolist(), ids.tolist()))
    assert (1, 1, 255) in set(zip(linhas.tolist(), colunas.tolist(), ids.tolist()))


def test_janela_climatica_nao_usa_dados_futuros():
    datas = pd.date_range("2025-01-01", "2025-01-15", tz="UTC")
    clima = pd.DataFrame({"lat_power": -10.0, "lon_power": -45.0, "data_utc": datas})
    for nome in ["T2M", "RH2M", "PRECTOTCORR", "WS10M", "GWETROOT"]:
        clima[nome] = np.arange(1, len(datas) + 1, dtype=float)
    saida = clima_janelas(clima).set_index("data_utc")
    dia = pd.Timestamp("2025-01-11", tz="UTC")
    assert saida.loc[dia, "clima_T2M"] == pytest.approx(4.5)
    assert saida.loc[dia, "clima_PRECTOTCORR"] == pytest.approx(36.0)
    clima.loc[clima.data_utc > pd.Timestamp("2025-01-08", tz="UTC"), "T2M"] = 999
    assert clima_janelas(clima).set_index("data_utc").loc[dia, "clima_T2M"] == pytest.approx(4.5)


def test_historico_preserva_positivos_parciais_e_bloqueia_producao_tardia():
    datas = pd.date_range("2025-01-01", "2025-01-08", tz="UTC")
    dados = pd.DataFrame({"grid_id": "a", "data_utc": datas, "evidencia_foco": False,
                          "mascara_completa": True, "producao_utc": datas + pd.Timedelta(days=1)})
    dados.loc[2, "evidencia_foco"] = True
    dados.loc[2, "mascara_completa"] = False
    saida = adicionar_historico_mascaras(dados).set_index("data_utc")
    assert saida.loc[pd.Timestamp("2025-01-05", tz="UTC"), "mascara_foco_2d"] == 1
    dados.loc[2, "producao_utc"] = pd.Timestamp("2025-01-05T00:01:00Z")
    saida = adicionar_historico_mascaras(dados).set_index("data_utc")
    assert pd.isna(saida.loc[pd.Timestamp("2025-01-05", tz="UTC"), "mascara_foco_2d"])
    dados.loc[2, "evidencia_foco"] = False
    dados.loc[2, "producao_utc"] = pd.Timestamp("2025-01-04T12:00:00Z")
    saida = adicionar_historico_mascaras(dados).set_index("data_utc")
    assert pd.isna(saida.loc[pd.Timestamp("2025-01-05", tz="UTC"), "mascara_foco_2d"])


def test_vegetacao_so_entra_depois_do_fim_e_da_publicacao():
    alvo = pd.DataFrame({"grid_id": ["a", "a"], "data_utc": pd.to_datetime(["2025-01-20", "2025-01-22"], utc=True)})
    vegetacao = pd.DataFrame({"grid_id": ["a", "a"], "composto_fim": ["2025-01-15", "2025-01-19"],
                              "disponivel_utc": ["2025-01-18T12:00:00Z", "2025-01-20T04:00:00Z"],
                              "ndvi": [0.4, 0.8], "evi": [0.3, 0.6], "frac_qualidade": [0.9, 0.8]})
    vegetacao["disponivel_utc"] = pd.to_datetime(vegetacao["disponivel_utc"], utc=True).astype("datetime64[ms, UTC]")
    saida = adicionar_vegetacao(alvo, vegetacao).sort_values("data_utc")
    assert saida["vegetacao_ndvi"].tolist() == pytest.approx([0.4, 0.8])
    vegetacao.loc[0, "disponivel_utc"] = "2025-01-15T12:00:00Z"
    with pytest.raises(ValueError, match="antes de terminar"):
        adicionar_vegetacao(alvo, vegetacao)


def test_firms_valida_janela_e_resposta_sem_inventar_deteccao():
    colunas = b"latitude,longitude,acq_date,acq_time,satellite,instrument,confidence,version,frp\n"
    vazio = ler_csv(colunas, "VIIRS_NOAA20_SP", date(2025, 1, 1), 5)
    assert vazio.empty
    linha = b"-10,-45,2025-01-03,0307,N20,VIIRS,n,2.0,7.5\n"
    dados = ler_csv(colunas + linha, "VIIRS_NOAA20_SP", date(2025, 1, 1), 5)
    assert dados.iloc[0]["instante_utc"] == pd.Timestamp("2025-01-03T03:07:00Z")
    with pytest.raises(ValueError):
        ler_csv(colunas + linha, "VIIRS_NOAA20_SP", date(2025, 1, 4), 2)
    assert list(segmentos(date(2025, 1, 1), date(2025, 1, 12))) == [
        (date(2025, 1, 1), 5), (date(2025, 1, 6), 5), (date(2025, 1, 11), 2)]


def test_particoes_nao_usam_rotulos_publicados_apos_a_proxima_emissao():
    linhas = []
    for dia in pd.date_range("2025-01-01", "2025-01-08", tz="UTC"):
        for coluna in [0, 12]:
            for i in range(12):
                linhas.append({"grid_id": f"{coluna}_{i}", "data_utc": dia, "coluna": coluna,
                               "mascara_completa": True, "alvo_condicional": i % 2,
                               "producao_utc": dia + pd.Timedelta(days=1, hours=12)})
    dados = pd.DataFrame(linhas)
    partes = dividir(dados, "2025-01-02", "2025-01-04", "2025-01-06", "2025-01-08")
    assert partes["treino"]["data_utc"].max() == pd.Timestamp("2025-01-01", tz="UTC")
    assert partes["calibracao"]["data_utc"].max() == pd.Timestamp("2025-01-03", tz="UTC")
    assert partes["decisao"]["data_utc"].max() == pd.Timestamp("2025-01-05", tz="UTC")
    assert set(partes["treino"]["grid_id"]).isdisjoint(partes["teste_espacial"]["grid_id"])
