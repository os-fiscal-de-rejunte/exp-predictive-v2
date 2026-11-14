import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, log_loss, precision_recall_curve, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from modelo.caracteristicas_grade import BASE, CLIMA, HISTORICO_FIRMS, HISTORICO_MASCARA, RELEVO, VEGETACAO
from modelo.coletar_mascaras import PASTA, sha256
from modelo.config import RAIZ

SAIDA = RAIZ / "modelo" / "artefatos" / "evolucao_grade"
MODOS = {
    "com_geografia": BASE + ["clima_" + c for c in CLIMA] + HISTORICO_MASCARA + HISTORICO_FIRMS + VEGETACAO + RELEVO,
    "sem_vegetacao": BASE + ["clima_" + c for c in CLIMA] + HISTORICO_MASCARA + HISTORICO_FIRMS + RELEVO,
    "sem_relevo": BASE + ["clima_" + c for c in CLIMA] + HISTORICO_MASCARA + HISTORICO_FIRMS + VEGETACAO,
    "sem_geografia": BASE + ["clima_" + c for c in CLIMA] + HISTORICO_MASCARA + HISTORICO_FIRMS,
    "sem_firms": BASE + ["clima_" + c for c in CLIMA] + HISTORICO_MASCARA + VEGETACAO + RELEVO,
    "firms_sem_mascara": BASE + ["clima_" + c for c in CLIMA] + HISTORICO_FIRMS + VEGETACAO + RELEVO,
    "clima_geografia": BASE + ["clima_" + c for c in CLIMA] + VEGETACAO + RELEVO,
    "somente_clima": BASE + ["clima_" + c for c in CLIMA],
}


def validar(dados):
    exigidas = {"grid_id", "data_utc", "producao_utc", "linha", "coluna", "mascara_completa", "alvo_condicional"}
    exigidas.update(c for colunas in MODOS.values() for c in colunas)
    ausentes = sorted(exigidas - set(dados.columns))
    if ausentes:
        raise ValueError(f"Colunas ausentes: {ausentes}")
    dados = dados.copy()
    dados["data_utc"] = pd.to_datetime(dados["data_utc"], utc=True, errors="raise")
    dados["producao_utc"] = pd.to_datetime(dados["producao_utc"], utc=True, errors="coerce")
    if dados.duplicated(["grid_id", "data_utc"]).any():
        raise ValueError("Celulas e datas duplicadas.")
    if dados["mascara_completa"].isna().any() or not dados["mascara_completa"].isin([True, False]).all():
        raise ValueError("Indicador de observacao invalido.")
    completos = dados["mascara_completa"].astype(bool)
    if dados.loc[~completos, "alvo_condicional"].notna().any():
        raise ValueError("Rotulo informado para mascara incompleta.")
    if not dados.loc[completos, "alvo_condicional"].isin([0, 1]).all():
        raise ValueError("Rotulo de mascara completa invalido.")
    for coluna in set(c for colunas in MODOS.values() for c in colunas):
        dados[coluna] = pd.to_numeric(dados[coluna], errors="raise")
        if np.isinf(dados[coluna].to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError(f"Valor infinito em {coluna}.")
    return dados.sort_values(["data_utc", "grid_id"]).reset_index(drop=True)


def dividir(dados, fim_treino, fim_calibracao, fim_decisao, fim_teste):
    limites = [pd.Timestamp(x, tz="UTC") for x in [fim_treino, fim_calibracao, fim_decisao, fim_teste]]
    if not all(a < b for a, b in zip(limites, limites[1:])):
        raise ValueError("Cortes temporais devem estar em ordem crescente.")
    d = dados[dados["mascara_completa"]].copy()
    data = d["data_utc"]
    local = d["coluna"] <= 9
    espacial = d["coluna"] >= 12
    treino, cal, decisao, teste = limites
    partes = {
        "treino": d[local & (data <= treino) & d["producao_utc"].le(treino + pd.Timedelta(hours=23))].copy(),
        "calibracao": d[local & (data > treino) & (data <= cal) & d["producao_utc"].le(cal + pd.Timedelta(hours=23))].copy(),
        "decisao": d[local & (data > cal) & (data <= decisao) & d["producao_utc"].le(decisao + pd.Timedelta(hours=23))].copy(),
        "teste_temporal": d[local & (data > decisao) & (data <= teste)].copy(),
        "teste_espacial": d[espacial & (data > decisao) & (data <= teste)].copy(),
    }
    for nome, parte in partes.items():
        contagem = parte["alvo_condicional"].value_counts()
        if len(contagem) != 2 or contagem.min() < 5:
            raise ValueError(f"Particao {nome} tem classes insuficientes: {contagem.to_dict()}.")
    return partes


def candidatos():
    return {
        "logistica": Pipeline([
            ("imputar", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("escalar", StandardScaler()),
            ("modelo", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "xgboost": Pipeline([
            ("imputar", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("modelo", XGBClassifier(n_estimators=120, max_depth=3, learning_rate=0.06,
                                     min_child_weight=12, subsample=0.85, colsample_bytree=0.9,
                                     reg_lambda=3, objective="binary:logistic", eval_metric="logloss",
                                     tree_method="hist", n_jobs=2, random_state=42)),
        ]),
    }


def selecionar(treino, colunas):
    datas = np.sort(treino["data_utc"].unique())
    if len(datas) < 12:
        raise ValueError("Treino precisa de ao menos 12 datas para comparacao cronologica.")
    grupos = np.array_split(datas, 4)
    resultados = {}
    for nome, modelo in candidatos().items():
        dobras = []
        for i in range(1, 4):
            ajuste = treino[treino["data_utc"].isin(np.concatenate(grupos[:i]))]
            validacao = treino[treino["data_utc"].isin(grupos[i])]
            ajuste = ajuste[ajuste["producao_utc"] <= validacao["data_utc"].min() - pd.Timedelta(hours=1)]
            if ajuste["alvo_condicional"].nunique() != 2 or validacao["alvo_condicional"].nunique() != 2:
                raise ValueError(f"Classes insuficientes na dobra {i} de {nome}.")
            modelo.fit(ajuste[colunas], ajuste["alvo_condicional"].astype(int))
            p = modelo.predict_proba(validacao[colunas])[:, 1]
            dobras.append({"inicio": validacao["data_utc"].min().date().isoformat(),
                           "fim": validacao["data_utc"].max().date().isoformat(),
                           "linhas": len(validacao), "positivos": int(validacao["alvo_condicional"].sum()),
                           "brier": float(brier_score_loss(validacao["alvo_condicional"].astype(int), p)),
                           "average_precision": float(average_precision_score(validacao["alvo_condicional"].astype(int), p))})
        resultados[nome] = {"brier_medio": float(np.mean([x["brier"] for x in dobras])), "dobras": dobras}
    nome = min(resultados, key=lambda n: resultados[n]["brier_medio"])
    modelo = candidatos()[nome]
    modelo.fit(treino[colunas], treino["alvo_condicional"].astype(int))
    return nome, modelo, resultados


def logito(prob):
    p = np.clip(np.asarray(prob), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def probabilidade(modelo, calibrador, dados, colunas):
    p = modelo.predict_proba(dados[colunas])[:, 1]
    return calibrador.predict_proba(logito(p))[:, 1] if calibrador is not None else p


def limiar_decisao(y, prob):
    precisao, revocacao, limiares = precision_recall_curve(y, prob)
    candidatos_validos = np.flatnonzero(precisao[:-1] >= 0.30)
    if len(candidatos_validos):
        melhor = candidatos_validos[np.argmax(revocacao[:-1][candidatos_validos])]
        return float(limiares[melhor]), "maior revocacao com precisao minima de 30% na decisao"
    soma = precisao[:-1] + revocacao[:-1]
    f1 = np.divide(2 * precisao[:-1] * revocacao[:-1], soma, out=np.zeros_like(soma), where=soma != 0)
    return float(limiares[np.argmax(f1)]), "F1 maximo; precisao minima nao atingida na decisao"


def metricas(y, p, limiar):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    previsto = (p >= limiar).astype(int)
    dois = len(np.unique(y)) == 2
    return {"linhas": len(y), "positivos": int(y.sum()), "prevalencia": float(y.mean()),
            "acuracia": float(accuracy_score(y, previsto)), "acuracia_balanceada": float(balanced_accuracy_score(y, previsto)),
            "precisao": float(precision_score(y, previsto, zero_division=0)),
            "revocacao": float(recall_score(y, previsto, zero_division=0)), "f1": float(f1_score(y, previsto, zero_division=0)),
            "average_precision": float(average_precision_score(y, p)) if dois else None,
            "roc_auc": float(roc_auc_score(y, p)) if dois else None,
            "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "matriz_confusao": confusion_matrix(y, previsto, labels=[0, 1]).tolist()}


def referencia_sazonal(treino, dados):
    media = float(treino["alvo_condicional"].mean())
    grupo = treino.groupby(treino["data_utc"].dt.month)["alvo_condicional"].agg(["sum", "count"])
    mensal = (grupo["sum"] + 20 * media) / (grupo["count"] + 20)
    return dados["data_utc"].dt.month.map(mensal).fillna(media).to_numpy(dtype=float)


def referencia_persistencia(treino, dados):
    media = float(treino["alvo_condicional"].mean())
    grupo = treino.groupby("mascara_foco_2d")["alvo_condicional"].agg(["sum", "count"])
    valores = (grupo["sum"] + 20 * media) / (grupo["count"] + 20)
    return dados["mascara_foco_2d"].map(valores).fillna(media).to_numpy(dtype=float)


def intervalo_blocos(dados, prob, baseline, limiar, repeticoes=300):
    y = dados["alvo_condicional"].astype(int).to_numpy()
    p = np.asarray(prob)
    b = np.asarray(baseline)
    quadro = pd.DataFrame({"dia": dados["data_utc"].to_numpy(), "positivos": y,
                           "tp": (p >= limiar) & (y == 1), "acertos": (p >= limiar) == y,
                           "erro": (p-y) ** 2, "erro_base": (b-y) ** 2})
    blocos = quadro.groupby("dia").agg({"positivos": "sum", "tp": "sum", "acertos": "sum", "erro": ["sum", "count"], "erro_base": "sum"})
    positivos = blocos[("positivos", "sum")].to_numpy(dtype=float)
    tp = blocos[("tp", "sum")].to_numpy(dtype=float)
    acertos = blocos[("acertos", "sum")].to_numpy(dtype=float)
    erro = blocos[("erro", "sum")].to_numpy(dtype=float)
    n = blocos[("erro", "count")].to_numpy(dtype=float)
    erro_base = blocos[("erro_base", "sum")].to_numpy(dtype=float)
    rng = np.random.default_rng(42)
    amostras = rng.integers(0, len(blocos), size=(repeticoes, len(blocos)))
    total = n[amostras].sum(axis=1)
    pos = positivos[amostras].sum(axis=1)
    recalls = np.divide(tp[amostras].sum(axis=1), pos, out=np.full(repeticoes, np.nan), where=pos > 0)
    acuracias = acertos[amostras].sum(axis=1) / total
    diferenca = (erro_base[amostras].sum(axis=1) - erro[amostras].sum(axis=1)) / total
    return {"metodo": "reamostragem de dias inteiros; nao elimina dependencia entre dias ou regioes",
            "blocos": len(blocos), "repeticoes": repeticoes,
            "acuracia_ic95": np.nanquantile(acuracias, [0.025, 0.975]).tolist(),
            "revocacao_ic95": np.nanquantile(recalls, [0.025, 0.975]).tolist(),
            "brier_sazonal_menos_modelo_ic95": np.nanquantile(diferenca, [0.025, 0.975]).tolist()}


def treinar(caminho, saida, fim_treino, fim_calibracao, fim_decisao, fim_teste):
    caminho, saida = Path(caminho), Path(saida)
    dados = validar(pd.read_parquet(caminho))
    partes = dividir(dados, fim_treino, fim_calibracao, fim_decisao, fim_teste)
    treino, cal, decisao = (partes[n] for n in ["treino", "calibracao", "decisao"])
    teste = pd.concat([partes["teste_temporal"], partes["teste_espacial"]]).sort_values(["data_utc", "grid_id"]).copy()
    sazonal = referencia_sazonal(treino, teste)
    persistencia = referencia_persistencia(treino, teste)
    relatorio = {
        "criado_em_utc": datetime.now(timezone.utc).isoformat(), "sha256_dados": sha256(caminho),
        "objetivo": "Ao menos um pixel VIIRS NOAA-20 com foco em celula de 0,25 grau no dia UTC, condicionado a mascara integralmente classificada.",
        "status": "experimental_retrospectivo_nao_operacional",
        "particoes": {nome: {"linhas": len(p), "positivos": int(p["alvo_condicional"].sum()),
                              "inicio": p["data_utc"].min().date().isoformat(), "fim": p["data_utc"].max().date().isoformat(),
                              "celulas": int(p["grid_id"].nunique())} for nome, p in partes.items()},
        "colunas_espaciais": {"treino": "0 a 9", "buffer_excluido": "10 e 11", "teste_reservado": "12 a 15"},
        "referencias": {"sazonal": metricas(teste["alvo_condicional"], sazonal, 0.5),
                        "persistencia": metricas(teste["alvo_condicional"], persistencia, 0.5)},
        "modos": {}, "bibliotecas": {n: version(n) for n in ["numpy", "pandas", "scikit-learn", "joblib", "xgboost"]},
        "limites": ["Datas historicas ja examinadas; nao constituem teste cego prospectivo.",
                    "A regiao piloto e pequena e nao representa todo o Brasil.",
                    "Os dados POWER podem ter sido revisados e o FIRMS SP pode diferir do NRT publicado na epoca.",
                    "Mascaras incompletas sao desconhecidas para o alvo, mesmo quando ha evidencia parcial de foco."]}
    modelos = {}
    saidas = teste[["grid_id", "data_utc", "linha", "coluna", "latitude", "longitude", "alvo_condicional"]].copy()
    saidas["referencia_sazonal"] = sazonal
    saidas["referencia_persistencia"] = persistencia
    with threadpool_limits(limits=2):
        for modo, colunas in MODOS.items():
            print(f"Treinando modo {modo}", flush=True)
            nome, modelo, selecao = selecionar(treino, colunas)
            p_cal = probabilidade(modelo, None, cal, colunas)
            calibrador = LogisticRegression(max_iter=1000, random_state=42).fit(logito(p_cal), cal["alvo_condicional"].astype(int))
            p_dec = probabilidade(modelo, None, decisao, colunas)
            p_dec_cal = calibrador.predict_proba(logito(p_dec))[:, 1]
            b_bruto = float(brier_score_loss(decisao["alvo_condicional"].astype(int), p_dec))
            b_cal = float(brier_score_loss(decisao["alvo_condicional"].astype(int), p_dec_cal))
            if b_cal >= b_bruto:
                calibrador = None
            p_lim = p_dec_cal if calibrador is not None else p_dec
            limiar, criterio = limiar_decisao(decisao["alvo_condicional"].astype(int), p_lim)
            p_teste = probabilidade(modelo, calibrador, teste, colunas)
            saidas[f"prob_{modo}"] = p_teste
            saidas[f"classe_{modo}"] = (p_teste >= limiar).astype(int)
            grupos = {"geral": np.ones(len(teste), dtype=bool), "temporal": teste["coluna"].to_numpy() <= 9,
                      "espacial": teste["coluna"].to_numpy() >= 12}
            avaliacao = {nome_grupo: metricas(teste.loc[selecao_grupo, "alvo_condicional"], p_teste[selecao_grupo], limiar)
                         for nome_grupo, selecao_grupo in grupos.items()}
            por_mes = {}
            for mes in sorted(teste["data_utc"].dt.strftime("%Y-%m").unique()):
                selecao_mes = teste["data_utc"].dt.strftime("%Y-%m").eq(mes).to_numpy()
                por_mes[mes] = metricas(teste.loc[selecao_mes, "alvo_condicional"], p_teste[selecao_mes], limiar)
            relatorio["modos"][modo] = {"modelo": nome, "colunas": colunas, "selecao": selecao,
                                        "calibrado": calibrador is not None, "brier_decisao_bruto": b_bruto,
                                        "brier_decisao_calibrado": b_cal, "limiar": limiar, "criterio_limiar": criterio,
                                        "teste": avaliacao, "por_mes": por_mes,
                                        "intervalos_teste_geral": intervalo_blocos(teste, p_teste, sazonal, limiar)}
            modelos[modo] = {"modelo": modelo, "calibrador": calibrador, "colunas": colunas, "limiar": limiar}
    saida.mkdir(parents=True, exist_ok=True)
    joblib.dump({"versao": 2, "modos": modelos, "sha256_dados": sha256(caminho),
                 "status": relatorio["status"]}, saida / "modelo_grade.joblib")
    (saida / "relatorio_grade.json").write_text(json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")
    saidas.to_csv(saida / "previsoes_teste.csv", index=False, lineterminator="\r\n")
    return relatorio


def main():
    parser = argparse.ArgumentParser(description="Treina modelos para a grade regional VIIRS.")
    parser.add_argument("--dados", type=Path, default=PASTA / "caracteristicas_grade.parquet")
    parser.add_argument("--saida", type=Path, default=SAIDA)
    parser.add_argument("--fim-treino", default="2025-04-30")
    parser.add_argument("--fim-calibracao", default="2025-06-30")
    parser.add_argument("--fim-decisao", default="2025-08-31")
    parser.add_argument("--fim-teste", default="2025-12-31")
    args = parser.parse_args()
    resultado = treinar(args.dados, args.saida, args.fim_treino, args.fim_calibracao, args.fim_decisao, args.fim_teste)
    print(json.dumps({"referencias": resultado["referencias"], "modos": {n: m["teste"] for n, m in resultado["modos"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
