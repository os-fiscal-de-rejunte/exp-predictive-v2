import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from modelo.coletar_mascaras import PASTA, sha256
from modelo.treinar_grade import MODOS, SAIDA, candidatos, dividir, limiar_decisao, logito, metricas, probabilidade, validar
from modelo.vizinhanca import AnomaliasClima, COLUNAS, adicionar

DESTINO = SAIDA.parent / "evolucao_vizinhanca"


def modelos():
    resultado = candidatos()
    resultado["floresta"] = Pipeline([
        ("imputar", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("modelo", RandomForestClassifier(n_estimators=80, max_depth=9, min_samples_leaf=12,
                                          max_features=0.8, n_jobs=2, random_state=42)),
    ])
    for nome in resultado:
        resultado[nome].steps.insert(0, ("anomalias", AnomaliasClima()))
    return resultado


def comparar(treino, colunas):
    datas = np.sort(treino["data_utc"].unique())
    grupos = np.array_split(datas, 4)
    resultados = {}
    for nome, modelo in modelos().items():
        dobras = []
        for i in range(1, 4):
            ajuste = treino[treino["data_utc"].isin(np.concatenate(grupos[:i]))]
            validacao = treino[treino["data_utc"].isin(grupos[i])]
            ajuste = ajuste[ajuste["producao_utc"] <= validacao["data_utc"].min() - pd.Timedelta(hours=1)]
            if ajuste["alvo_condicional"].nunique() != 2 or validacao["alvo_condicional"].nunique() != 2:
                raise ValueError("Classes insuficientes em uma dobra cronologica.")
            modelo.fit(ajuste[colunas], ajuste["alvo_condicional"].astype(int))
            p = modelo.predict_proba(validacao[colunas])[:, 1]
            dobras.append({"inicio": validacao["data_utc"].min().date().isoformat(),
                           "fim": validacao["data_utc"].max().date().isoformat(), "linhas": len(validacao),
                           "positivos": int(validacao["alvo_condicional"].sum()),
                           "brier": float(brier_score_loss(validacao["alvo_condicional"].astype(int), p)),
                           "average_precision": float(average_precision_score(validacao["alvo_condicional"].astype(int), p))})
        resultados[nome] = {"brier_medio": float(np.mean([d["brier"] for d in dobras])), "dobras": dobras}
        print(f"Candidato {nome}: Brier medio {resultados[nome]['brier_medio']:.5f}", flush=True)
    escolhido = min(resultados, key=lambda x: resultados[x]["brier_medio"])
    modelo = modelos()[escolhido]
    modelo.fit(treino[colunas], treino["alvo_condicional"].astype(int))
    return escolhido, modelo, resultados


def treinar(caminho, saida):
    caminho, saida = Path(caminho), Path(saida)
    dados = adicionar(validar(pd.read_parquet(caminho)))
    partes = dividir(dados, "2025-04-30", "2025-06-30", "2025-08-31", "2025-12-31")
    treino, cal, decisao = (partes[n] for n in ["treino", "calibracao", "decisao"])
    teste = pd.concat([partes["teste_temporal"], partes["teste_espacial"]]).sort_values(["data_utc", "grid_id"])
    referencia = json.loads((SAIDA / "relatorio_grade.json").read_text(encoding="utf-8"))
    if referencia["sha256_dados"] != sha256(caminho):
        raise ValueError("A base diverge da referencia regional.")
    relatorio = {"criado_em_utc": datetime.now(timezone.utc).isoformat(), "sha256_dados": sha256(caminho),
                 "status": "pesquisa_retrospectiva_nao_operacional", "modos": {},
                 "bibliotecas": {n: version(n) for n in ["numpy", "pandas", "scikit-learn", "joblib", "xgboost"]},
                 "limites": "Apenas sinais anteriores sao agregados; FIRMS SP e clima historico podem ter sido revisados. "
                            "Teste ja explorado e mesma regiao; nao comprova melhoria prospectiva."}
    pacotes = {}
    with threadpool_limits(limits=2):
        for modo in ["com_geografia", "firms_sem_mascara"]:
            print(f"Comparando vizinhanca: {modo}", flush=True)
            colunas_viz = [x for x in COLUNAS if modo != "firms_sem_mascara" or not x.startswith("viz_mascara_")]
            colunas = MODOS[modo] + colunas_viz + ["mes"]
            nome, modelo, selecao = comparar(treino, colunas)
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
            relatorio["modos"][modo] = {"modelo": nome, "colunas": colunas, "selecao": selecao,
                                        "calibrado": calibrador is not None, "limiar": limiar,
                                        "criterio_limiar": criterio, "teste": avaliacao,
                                        "referencia_cv_brier": referencia["modos"][modo]["selecao"][referencia["modos"][modo]["modelo"]]["brier_medio"],
                                        "referencia_teste": referencia["modos"][modo]["teste"]}
            pacotes[modo] = {"modelo": modelo, "calibrador": calibrador, "colunas": colunas, "limiar": limiar}
    saida.mkdir(parents=True, exist_ok=True)
    joblib.dump({"modos": pacotes, "sha256_dados": sha256(caminho), "status": relatorio["status"]}, saida / "modelo_vizinhanca.joblib")
    (saida / "relatorio_vizinhanca.json").write_text(json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")
    return relatorio


def main():
    parser = argparse.ArgumentParser(description="Compara vizinhanca e anomalias sazonais com o modelo regional.")
    parser.add_argument("--dados", type=Path, default=PASTA / "caracteristicas_grade.parquet")
    parser.add_argument("--saida", type=Path, default=DESTINO)
    args = parser.parse_args()
    resultado = treinar(args.dados, args.saida)
    print(json.dumps({n: {"modelo": m["modelo"], "teste": m["teste"]} for n, m in resultado["modos"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
