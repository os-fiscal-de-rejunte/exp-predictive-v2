from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from modelo.mascaras import RAIO
from modelo.coletar_firms import ler_csv
from modelo.coletar_mascaras import descobrir
from modelo.testar_suica import adicionar_firms, avaliar_medidas, mapa_pixels


def test_mapeamento_suico_usa_celulas_selecionadas():
    grade = pd.DataFrame({"linha": [1], "coluna": [5]})
    latitude = 46.125
    y = np.array([RAIO * np.deg2rad(latitude)])
    x = RAIO * np.deg2rad(np.array([7.125, 7.375])) * np.cos(np.deg2rad(latitude))
    linhas, colunas, ids = mapa_pixels(x, y, grade)
    assert linhas.tolist() == [0]
    assert colunas.tolist() == [0]
    assert ids.tolist() == [0]


def test_firms_suico_respeita_celula_e_janela_temporal():
    grade = pd.DataFrame({"grid_id": ["a"], "linha": [1], "coluna": [5]})
    dias = pd.date_range("2025-01-01", "2025-01-10", tz="UTC")
    dados = pd.DataFrame({"grid_id": "a", "data_utc": dias})
    detecoes = pd.DataFrame({"latitude": [46.125, 46.125, 46.125], "longitude": [7.125, 7.125, 8.125],
                             "instante_utc": [datetime(2025, 1, 1, tzinfo=timezone.utc),
                                              datetime(2025, 1, 8, tzinfo=timezone.utc),
                                              datetime(2025, 1, 8, tzinfo=timezone.utc)],
                             "fonte": ["VIIRS_NOAA20_SP"] * 3, "frp": [20.0, 5.0, 10.0]})
    saida = adicionar_firms(dados, detecoes, grade)
    ultimo = saida[saida["data_utc"] == dias[-1]].iloc[0]
    assert ultimo["viirs_registros_7d"] == 1
    assert ultimo["viirs_frp_7d"] == 5.0
    assert np.isnan(ultimo["modis_registros_7d"])


def test_leitura_firms_aceita_area_suica_explicita():
    conteudo = (b"latitude,longitude,acq_date,acq_time,satellite,instrument,confidence,version,frp\n"
                b"46.125,7.125,2025-01-02,1200,N20,VIIRS,n,2.0,3.0\n")
    dados = ler_csv(conteudo, "VIIRS_NOAA20_SP", date(2025, 1, 1), 5, "5.75,45.75,10.75,47.75")
    assert len(dados) == 1
    assert dados.iloc[0]["latitude"] == 46.125


def test_catalogo_recebe_area_e_tile_suicos(tmp_path, monkeypatch):
    nome = "VJ114A1.A2025001.h18v04.002.2025003010203"
    class Resposta:
        headers = {"CMR-Hits": "1"}
        content = b"catalogo_de_teste"
        def raise_for_status(self):
            pass
        def json(self):
            return {"feed": {"entry": [{"title": nome, "links": [{"href": "https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/VJ114A1.002/" + nome + "/" + nome + ".h5"}]}]}}
    def requisitar(url, params, timeout):
        assert params["bounding_box"] == "5.75,45.75,10.75,47.75"
        return Resposta()
    monkeypatch.setattr("modelo.coletar_mascaras.requests.get", requisitar)
    itens = descobrir(date(2025, 1, 1), date(2025, 1, 1), tmp_path, "5.75,45.75,10.75,47.75", {"h18v04"})
    assert len(itens) == 1
    assert itens[0]["tile"] == "h18v04"


def test_metricas_nao_inventam_precisao_sem_positivos_previstos():
    resultado = avaliar_medidas([0, 1], [0.01, 0.02], [0, 0])
    assert resultado["precisao"] is None
    assert resultado["revocacao"] == 0
    assert resultado["matriz_confusao"] == [[1, 0], [1, 0]]
    sem_positivos = avaliar_medidas([0, 0], [0.01, 0.02], [0, 0])
    assert sem_positivos["revocacao"] is None
    assert sem_positivos["matriz_confusao"] == [[2, 0], [0, 0]]
