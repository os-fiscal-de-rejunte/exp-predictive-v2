import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from threadpoolctl import threadpool_limits

from modelo.coletar_mascaras import PASTA, sha256
from modelo.cobertura_solo import COLUNAS, adicionar
from modelo.treinar_grade import MODOS, SAIDA, dividir, limiar_decisao, logito, metricas, probabilidade, selecionar, validar

DESTINO = SAIDA.parent / "evolucao_cobertura_solo"


def treinar(caminho, cobertura, saida):
    caminho, cobertura, saida = map(Path, [caminho, cobertura, saida])
    resumo = json.loads((cobertura.parent / "resumo_cobertura_solo.json").read_text(encoding="utf-8"))
    if resumo["sha256_arquivo"] != sha256(cobertura):
        raise ValueError("Dados de cobertura do solo divergiram do manifesto.")
    dados = adicionar(validar(pd.read_parquet(caminho)), pd.read_parquet(cobertura))
    partes = dividir(dados, "2025-04-30", "2025-06-30", "2025-08-31", "2025-12-31")
    treino, cal, decisao = (partes[n] for n in ["treino", "calibracao", "decisao"])
    teste = pd.concat([partes["teste_temporal"], partes["teste_espacial"]]).sort_values(["data_utc", "grid_id"])
    referencia = json.loads((SAIDA / "relatorio_grade.json").read_text(encoding="utf-8"))
    if referencia["sha256_dados"] != sha256(caminho):
        raise ValueError("Base divergente do relatorio regional.")
    relatorio = {"criado_em_utc": datetime.now(timezone.utc).isoformat(), "sha256_dados": sha256(caminho),
                 "sha256_cobertura_solo": sha256(cobertura), "status": "pesquisa_retrospectiva_nao_operacional",
                 "modos": {}, "limites": "Cobertura IGBP agregada e datada conservadoramente. "
                 "As classes contem erros; o teste ja foi explorado e cobre uma unica regiao."}
    pacotes = {}
    with threadpool_limits(limits=2):
        for modo in ["com_geografia", "firms_sem_mascara"]:
            print(f"Comparando cobertura do solo: {modo}", flush=True)
            colunas = MODOS[modo] + COLUNAS
            nome, modelo, selecao = selecionar(treino, colunas)
            p_cal = probabilidade(modelo, None, cal, colunas)
            calibrador = LogisticRegression(max_iter=1000, random_state=42).fit(logito(p_cal), cal["alvo_condicional"].astype(int))
            p_dec = probabilidade(modelo, None, decisao, colunas)
            p_dec_cal = calibrador.predict_proba(logito(p_dec))[:, 1]
            if brier_score_loss(decisao["alvo_condicional"].astype(int), p_dec_cal) >= brier_score_loss(decisao["alvo_condicional"].astype(int), p_dec):
                calibrador = None
            p_lim = p_dec_cal if calibrador is not None else p_dec
            limiar, criterio = limiar_decisao(decisao["alvo_condicional"].astype(int), p_lim)
            p_teste = probabilidade(modelo, calibrador, teste, colunas)
            grupos = {"geral": np.ones(len(teste), dtype=bool), "temporal": teste["coluna"].to_numpy() <= 9,
                      "espacial": teste["coluna"].to_numpy() >= 12}
            avaliacao = {grupo: metricas(teste.loc[filtro, "alvo_condicional"], p_teste[filtro], limiar)
                         for grupo, filtro in grupos.items()}
            classes = teste["solo_classe_dominante"]
            tipos = {"floresta": classes.between(1, 5), "savana_campo": classes.between(6, 10),
                     "cultivo_ou_mosaico": classes.isin([12, 14]),
                     "outras": classes.notna() & ~classes.between(1, 10) & ~classes.isin([12, 14]),
                     "desconhecida": classes.isna()}
            por_cobertura = {}
            for grupo, filtro in tipos.items():
                selecao_grupo = filtro.to_numpy()
                if selecao_grupo.any():
                    medidas = metricas(teste.loc[filtro, "alvo_condicional"], p_teste[selecao_grupo], limiar)
                    medidas["status"] = "exploratorio" if medidas["positivos"] >= 50 else "positivos_insuficientes"
                    por_cobertura[grupo] = medidas
            relatorio["modos"][modo] = {"modelo": nome, "colunas": colunas, "selecao": selecao,
                                        "calibrado": calibrador is not None, "limiar": limiar,
                                        "criterio_limiar": criterio, "teste": avaliacao, "por_cobertura": por_cobertura,
                                        "referencia_cv_brier": referencia["modos"][modo]["selecao"][referencia["modos"][modo]["modelo"]]["brier_medio"],
                                        "referencia_teste": referencia["modos"][modo]["teste"]}
            pacotes[modo] = {"modelo": modelo, "calibrador": calibrador, "colunas": colunas, "limiar": limiar}
    saida.mkdir(parents=True, exist_ok=True)
    joblib.dump({"modos": pacotes, "sha256_dados": sha256(caminho), "sha256_cobertura": sha256(cobertura),
                 "status": relatorio["status"]}, saida / "modelo_cobertura_solo.joblib")
    (saida / "relatorio_cobertura_solo.json").write_text(json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")
    return relatorio


def main():
    parser = argparse.ArgumentParser(description="Compara cobertura do solo com o modelo regional.")
    parser.add_argument("--dados", type=Path, default=PASTA / "caracteristicas_grade.parquet")
    parser.add_argument("--cobertura", type=Path, default=PASTA / "cobertura_solo_grade.parquet")
    parser.add_argument("--saida", type=Path, default=DESTINO)
    args = parser.parse_args()
    resultado = treinar(args.dados, args.cobertura, args.saida)
    print(json.dumps({n: {"modelo": m["modelo"], "teste": m["teste"]} for n, m in resultado["modos"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
