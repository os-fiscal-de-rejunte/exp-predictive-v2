from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modelo.coletar_mascaras import sha256
from modelo.atualizar_grade import pasta_geografia
from modelo.preparar_grade_atual import estender_clima
from modelo.prever_grade import geojson, prever
from modelo.registro_prospectivo import carregar, registrar
from modelo.avaliar_prospectivo import avaliar, selecionar_registros
from modelo.treinar_grade import SAIDA as ARTEFATOS


def test_clima_atual_nao_inventa_valores_futuros():
    datas = pd.date_range("2026-11-05", "2026-11-12", tz="UTC")
    dados = pd.DataFrame({"lat_power": -10.0, "lon_power": -45.0, "data_utc": datas,
                          "T2M": 30.0, "RH2M": 50.0, "PRECTOTCORR": 1.0,
                          "WS10M": 2.0, "GWETROOT": 0.3})
    saida = estender_clima(dados, datetime(2026, 11, 15, tzinfo=timezone.utc).date())
    futuro = saida[saida["data_utc"] > datas.max()]
    assert len(futuro) == 3
    assert futuro[["T2M", "RH2M", "PRECTOTCORR", "WS10M", "GWETROOT"]].isna().all().all()
    dados.loc[0, "data_utc"] = pd.Timestamp("2026-11-13", tz="UTC")
    with pytest.raises(ValueError, match="posteriores"):
        estender_clima(dados, datetime(2026, 11, 15, tzinfo=timezone.utc).date())


def test_geojson_mantem_condicao_e_geometria_fechada():
    resultado = {"status": "experimental", "interpretacao": "condicional", "celulas": [
        {"celula_id": "a", "limites": [-47, -12, -46.75, -11.75], "status": "experimental",
         "modo_modelo": "firms_sem_mascara", "probabilidade_condicional": 0.2}]}
    mapa = geojson(resultado)
    coordenadas = mapa["features"][0]["geometry"]["coordinates"][0]
    assert coordenadas[0] == coordenadas[-1]
    assert mapa["interpretacao"] == "condicional"


def test_geografia_nova_nao_depende_de_arquivos_brutos_historicos(tmp_path):
    historica, atual = tmp_path / "historica", tmp_path / "atual"
    historica.mkdir()
    (historica / "manifesto_geografia.json").write_text(json.dumps({"arquivos": {"a.h5": {}}}), encoding="utf-8")
    assert pasta_geografia(historica, atual) == atual / "geografia"
    brutos = historica / "brutos" / "geografia"
    brutos.mkdir(parents=True)
    (brutos / "a.h5").write_bytes(b"teste")
    assert pasta_geografia(historica, atual) == historica
    (atual / "geografia").mkdir(parents=True)
    assert pasta_geografia(historica, atual) == atual / "geografia"


def test_previsao_real_recusa_falta_de_clima_e_arquivo_alterado(tmp_path, monkeypatch):
    pasta = Path("dados/evolucao_atual")
    candidatos = sorted(pasta.glob("caracteristicas_atuais_*.parquet"))
    modelo, relatorio = ARTEFATOS / "modelo_grade.joblib", ARTEFATOS / "relatorio_grade.json"
    if not candidatos or not modelo.exists() or not relatorio.exists():
        pytest.skip("Dados e artefato reais ainda nao gerados.")
    fonte = candidatos[-1]
    meta = json.loads(fonte.with_suffix(".json").read_text(encoding="utf-8"))
    instante = pd.Timestamp(meta["preparado_em_utc"]).to_pydatetime()
    class Relogio(datetime):
        @classmethod
        def now(cls, tz=None):
            return instante.astimezone(tz or timezone.utc)
    monkeypatch.setattr("modelo.prever_grade.datetime", Relogio)
    dados = pd.read_parquet(fonte)
    alvo = pd.Timestamp(meta["alvo_utc"], tz="UTC")
    linha = dados.index[dados["data_utc"].eq(alvo)][0]
    celula = dados.loc[linha, "grid_id"]
    dados.loc[linha, "clima_T2M"] = np.nan
    copia = tmp_path / "entrada.parquet"
    dados.to_parquet(copia, index=False)
    meta["sha256_saida"] = sha256(copia)
    copia.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
    resposta = prever(copia, modelo, relatorio)
    item = next(x for x in resposta["celulas"] if x["celula_id"] == celula)
    assert item["status"] == "dados_insuficientes"
    assert item["probabilidade_condicional"] is None
    assert resposta["probabilidade_incendio_incondicional"] is None
    meta["sha256_saida"] = "invalido"
    copia.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="divergiram"):
        prever(copia, modelo, relatorio)


def test_registro_prospectivo_preserva_emissao_e_recusa_alteracao(tmp_path):
    agora = datetime.now(timezone.utc)
    inicio = datetime.combine(agora.date() + timedelta(days=1), datetime.min.time(), timezone.utc)
    resultado = {"emitido_em_utc": agora.isoformat(), "inicio_previsto_utc": inicio.isoformat(),
                 "fim_previsto_utc": (inicio + timedelta(days=1)).isoformat(), "sha256_modelo": "abc",
                 "celulas": [{"celula_id": "a", "status": "experimental", "modo_modelo": "m",
                              "probabilidade_condicional": 0.7, "limiar_classificacao": 0.2, "classe_detectada": 1}]}
    caminho = registrar(resultado, tmp_path)
    assert carregar(caminho) == resultado
    assert len(selecionar_registros(tmp_path)) == 1
    caminho.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="alterado"):
        carregar(caminho)


def test_avaliacao_prospectiva_preserva_mascara_incompleta(tmp_path, monkeypatch):
    agora = datetime.now(timezone.utc)
    inicio = datetime.combine(agora.date() + timedelta(days=1), datetime.min.time(), timezone.utc)
    resultado = {"emitido_em_utc": agora.isoformat(), "inicio_previsto_utc": inicio.isoformat(),
                 "fim_previsto_utc": (inicio + timedelta(days=1)).isoformat(), "sha256_modelo": "abc",
                 "celulas": [{"celula_id": nome, "status": "experimental", "modo_modelo": "m",
                              "probabilidade_condicional": 0.7, "limiar_classificacao": 0.2, "classe_detectada": 1}
                             for nome in ["a", "b"]]}
    registrar(resultado, tmp_path)
    registros = selecionar_registros(tmp_path)
    assert avaliar(registros, agora=agora)["dias"][0]["status"] == "aguardando_fim_do_dia"
    observacoes = pd.DataFrame({"data_utc": [inicio, inicio], "grid_id": ["a", "b"],
                               "mascara_completa": [True, False], "alvo_condicional": [1, None],
                               "evidencia_foco": [True, True]})
    monkeypatch.setattr("modelo.avaliar_prospectivo.ler_mascaras", lambda _: observacoes)
    avaliacao = avaliar(registros, "mascaras.parquet", agora=inicio + timedelta(days=2))
    assert avaliacao["dias"][0]["comparadas"] == 1
    assert avaliacao["dias"][0]["observacoes_desconhecidas"] == 1
    assert avaliacao["dias"][0]["focos_em_mascara_incompleta"] == 1
    assert avaliacao["grupos"][0]["status"] == "positivos_insuficientes"
    assert avaliacao["grupos"][0]["revocacao"] == 1.0
