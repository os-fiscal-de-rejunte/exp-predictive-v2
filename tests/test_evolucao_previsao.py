from datetime import date

import joblib
import numpy as np
import pandas as pd
import pytest
import requests

from modelo.historico_firms import RAIO, disponibilidade, dentro, limites
from modelo.prever_atual import prever_atual
from modelo.viirs_dados import ARTEFATOS, BASE, CARACTERISTICAS, ler, sha256


def test_filtro_respeita_a_geometria_sinusoidal():
    geometria = {"xllcorner": -5000000.0, "yllcorner": -1200000.0,
                 "ncols": 40, "nrows": 40, "cellsize_m": 1000.0}
    y = np.array([-1180000.0, -1180000.0])
    x = np.array([-4980000.0, -4900000.0])
    lat = np.rad2deg(y / RAIO)
    lon = np.rad2deg(x / (RAIO * np.cos(np.deg2rad(lat))))
    dados = pd.DataFrame({"latitude": lat, "longitude": lon})
    assert len(dentro(dados, geometria)) == 1
    oeste, sul, leste, norte = limites(geometria)
    assert oeste < lon[0] < leste and sul < lat[0] < norte


def test_erro_de_firms_nao_expoe_chave(monkeypatch):
    def falhar(*args, **kwargs):
        raise requests.ConnectionError("chave_de_teste_sensivel")
    monkeypatch.setattr("modelo.historico_firms.requests.get", falhar)
    with pytest.raises(RuntimeError) as erro:
        disponibilidade("chave_de_teste_sensivel", "VIIRS_NOAA20_NRT")
    assert "chave_de_teste_sensivel" not in str(erro.value)


def test_previsao_escolhe_modelo_compativel_com_historico(tmp_path, monkeypatch):
    anterior = joblib.load(ARTEFATOS / "modelo_viirs.joblib")
    registro = ler(BASE).iloc[0]
    regiao = {k: getattr(registro, k) for k in ["region_id", "region_name", "latitude", "longitude", "spatial_group",
                                                "xllcorner", "yllcorner", "ncols", "nrows", "cellsize_m"]}
    pacote = {"modelo": anterior["modelo"], "calibrador": anterior["calibrador"], "limiar": anterior["limiar"]}
    artefato = {"versao": 2, "caracteristicas": CARACTERISTICAS, "sha256_dados": sha256(BASE),
                "regioes": [regiao], "modos": {"viirs_firms": pacote, "sem_satelite": pacote}}
    caminho = tmp_path / "modelo.joblib"
    joblib.dump(artefato, caminho)
    clima = {"weather_T2M": 26.0, "weather_RH2M": 60.0, "weather_PRECTOTCORR": 20.0,
             "weather_WS10M": 2.0, "weather_GWETROOT": 0.5}
    monkeypatch.setattr("modelo.prever_atual.consultar_clima", lambda *args: (clima, "a" * 64))
    monkeypatch.setattr("modelo.prever_atual.consultar_historico", lambda *args: {
        "viirs": {"situacao": "consultado", "registros_positivos": 3},
        "modis": {"situacao": "consultado", "registros_positivos": None},
    })
    resposta = prever_atual(BASE, caminho, "Brasil")
    assert resposta["recortes"][0]["modo_modelo"] == "viirs_firms"
    assert resposta["probabilidade_incendio_incondicional"] is None
    def falhar(*args):
        raise RuntimeError("Fonte indisponivel")
    monkeypatch.setattr("modelo.prever_atual.consultar_historico", falhar)
    resposta = prever_atual(BASE, caminho, "Brasil")
    assert resposta["recortes"][0]["modo_modelo"] == "sem_satelite"
