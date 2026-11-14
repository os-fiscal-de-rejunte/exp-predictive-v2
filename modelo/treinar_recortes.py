import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from modelo.coletar_mascaras import sha256
from modelo.config import RAIZ
from modelo.treinar_grade import limiar_decisao
from modelo.treinar_viirs import logito, metricas, particionar, probabilidades, referencia_sazonal, selecionar
from modelo.viirs_dados import BASE, CARACTERISTICAS, ler

SAIDA = RAIZ / "modelo" / "artefatos" / "evolucao_recortes"


def preparar_modo(dados, modo):
    saida = dados.copy()
    saida[["past_modis_event", "past_viirs_mask_event"]] = np.nan
    if modo == "sem_satelite":
        saida["past_viirs_log_records"] = np.nan
    elif modo != "viirs_firms":
        raise ValueError("Modo de historico desconhecido.")
    return saida


def treinar(caminho=BASE, saida=SAIDA):
    caminho, saida = Path(caminho), Path(saida)
    original = ler(caminho)
    relatorio = {"criado_em_utc": datetime.now(timezone.utc).isoformat(), "sha256_dados": sha256(caminho),
                 "status": "experimental_retrospectivo_nao_operacional", "modos": {},
                 "limites": ["Amostra de oito recortes selecionados e historico nao continuo.",
                             "Os dados de teste ja foram examinados anteriormente; nao sao prospectivos.",
                             "FIRMS padrao historico pode diferir do produto NRT atual.",
                             "Nao confundir ausencia de registros com ausencia de foco."]}
    modelos = {}
    for modo in ["viirs_firms", "sem_satelite"]:
        partes = particionar(preparar_modo(original, modo))
        treino, cal, decisao = (partes[n] for n in ["treino", "calibracao", "decisao"])
        nome, modelo, selecao = selecionar(treino)
        p_cal = probabilidades(modelo, None, cal)
        calibrador = LogisticRegression(max_iter=1000, random_state=42).fit(logito(p_cal), cal["event"].astype(int))
        p_dec = probabilidades(modelo, None, decisao)
        p_dec_cal = calibrador.predict_proba(logito(p_dec))[:, 1]
        bruto = float(brier_score_loss(decisao["event"].astype(int), p_dec))
        calibrado = float(brier_score_loss(decisao["event"].astype(int), p_dec_cal))
        if calibrado >= bruto:
            calibrador = None
        p_decisao = p_dec_cal if calibrador is not None else p_dec
        limiar, criterio = limiar_decisao(decisao["event"].astype(int), p_decisao)
        teste = pd.concat([partes["teste_local"], partes["teste_espacial"]]).sort_values(["period_start", "region_id"]).copy()
        p_teste = probabilidades(modelo, calibrador, teste)
        grupos = {"geral": np.ones(len(teste), dtype=bool), "areas_ajuste": teste["spatial_group"].eq("train").to_numpy(),
                  "areas_reservadas": teste["spatial_group"].eq("holdout").to_numpy()}
        avaliacao = {n: metricas(teste.loc[s, "event"], p_teste[s], limiar) for n, s in grupos.items()}
        sazonal = referencia_sazonal(treino, teste)
        relatorio["modos"][modo] = {"modelo": nome, "limiar": limiar, "criterio_limiar": criterio,
                                    "calibrado": calibrador is not None, "positivos_calibracao": int(cal["event"].sum()),
                                    "positivos_decisao": int(decisao["event"].sum()),
                                    "brier_decisao_bruto": bruto, "brier_decisao_calibrado": calibrado,
                                    "selecao": selecao, "teste": avaliacao,
                                    "referencia_sazonal": metricas(teste["event"], sazonal, 0.5)}
        modelos[modo] = {"modelo": modelo, "calibrador": calibrador, "limiar": limiar}
    regioes = original[["region_id", "region_name", "latitude", "longitude", "spatial_group", "xllcorner", "yllcorner", "ncols", "nrows", "cellsize_m"]].drop_duplicates()
    if regioes["region_id"].duplicated().any():
        raise ValueError("Geometria variavel para uma mesma regiao.")
    saida.mkdir(parents=True, exist_ok=True)
    joblib.dump({"versao": 2, "modos": modelos, "caracteristicas": CARACTERISTICAS,
                 "sha256_dados": sha256(caminho), "regioes": regioes.to_dict("records"),
                 "avaliacao": {n: d["teste"]["geral"] for n, d in relatorio["modos"].items()},
                 "referencias": {n: d["referencia_sazonal"] for n, d in relatorio["modos"].items()}}, saida / "modelo_recortes.joblib")
    (saida / "relatorio_recortes.json").write_text(json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")
    return relatorio


def main():
    parser = argparse.ArgumentParser(description="Treina modos separados com e sem historico FIRMS.")
    parser.add_argument("--dados", type=Path, default=BASE)
    parser.add_argument("--saida", type=Path, default=SAIDA)
    args = parser.parse_args()
    resultado = treinar(args.dados, args.saida)
    print(json.dumps({n: d["teste"] for n, d in resultado["modos"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
