import pandas as pd
import pytest

from modelo.cobertura_solo import adicionar


def test_cobertura_so_entra_apos_publicacao():
    dados = pd.DataFrame({"grid_id": ["a", "a", "a"],
                          "data_utc": pd.to_datetime(["2024-02-01", "2024-09-08", "2024-09-10"], utc=True)})
    cobertura = pd.DataFrame({"grid_id": ["a", "a"], "solo_ano": [2022, 2023],
                              "solo_disponivel_utc": pd.to_datetime(["2023-09-05T12:00:00Z", "2024-09-08T12:00:00Z"], utc=True),
                              "solo_cultivo_frac": [0.2, 0.4]})
    saida = adicionar(dados, cobertura)
    assert saida["solo_ano"].tolist() == [2022, 2022, 2023]
    assert saida["solo_cultivo_frac"].tolist() == [0.2, 0.2, 0.4]
    cobertura.loc[1, "solo_disponivel_utc"] = pd.Timestamp("2023-09-01", tz="UTC")
    with pytest.raises(ValueError, match="antes do fim"):
        adicionar(dados, cobertura)
