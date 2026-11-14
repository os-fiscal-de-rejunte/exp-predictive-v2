import numpy as np
import pandas as pd

from modelo.vizinhanca import AnomaliasClima, adicionar


def test_vizinhos_nao_atravessam_bordas_e_preservam_ausencia():
    dados = pd.DataFrame([{"data_utc": pd.Timestamp("2025-01-01", tz="UTC"), "linha": i, "coluna": j,
                           "mascara_foco_2d": np.nan, "viirs_registros_7d": np.nan, "modis_registros_7d": np.nan}
                          for i in range(16) for j in range(16)])
    dados.loc[(dados["linha"] == 0) & (dados["coluna"] == 1), "viirs_registros_7d"] = 3.0
    dados.loc[(dados["linha"] == 0) & (dados["coluna"] == 15), "viirs_registros_7d"] = 20.0
    saida = adicionar(dados)
    canto = saida[(saida["linha"] == 0) & (saida["coluna"] == 0)].iloc[0]
    assert canto["viz_viirs_soma"] == 3.0
    assert canto["viz_viirs_com_registro"] == 1
    assert np.isnan(canto["viz_modis_soma"])
    assert canto["viz_modis_com_registro"] == 0


def test_anomalias_aprendem_somente_com_a_particao_de_ajuste():
    colunas = ["clima_T2M", "clima_RH2M", "clima_PRECTOTCORR", "clima_WS10M", "clima_GWETROOT"]
    ajuste = pd.DataFrame({"mes": [1, 1, 2], **{c: [1.0, 3.0, 10.0] for c in colunas}})
    futuro = pd.DataFrame({"mes": [1, 3], **{c: [5.0, 6.0] for c in colunas}})
    transformador = AnomaliasClima().fit(ajuste)
    resultado = transformador.transform(futuro)
    assert resultado.loc[0, "clima_T2M_anomalia"] == 3.0
    assert np.isclose(resultado.loc[1, "clima_T2M_anomalia"], 6.0 - (14.0 / 3.0))
