import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, log_loss, precision_recall_curve, precision_score, recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modelo.viirs_dados import ARTEFATOS, BASE, CARACTERISTICAS, caracteristicas, ler, sha256


def candidatos():
    def criar(modelo):
        return Pipeline([
            ("imputar", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("escalar", StandardScaler()),
            ("modelo", modelo),
        ])
    return {
        "frequencia": criar(DummyClassifier(strategy="prior")),
        "logistica": criar(LogisticRegression(max_iter=2000, random_state=42)),
        "floresta": criar(RandomForestClassifier(n_estimators=150, min_samples_leaf=6, random_state=42, n_jobs=2)),
    }


def metricas(y, prob, limiar):
    y = np.asarray(y, dtype=int)
    prob = np.asarray(prob, dtype=float)
    previsto = (prob >= limiar).astype(int)
    return {
        "exemplos": len(y), "positivos": int(y.sum()), "prevalencia": float(y.mean()),
        "acuracia": float(accuracy_score(y, previsto)),
        "acuracia_balanceada": float(balanced_accuracy_score(y, previsto)),
        "precisao": float(precision_score(y, previsto, zero_division=0)),
        "revocacao": float(recall_score(y, previsto, zero_division=0)),
        "f1": float(f1_score(y, previsto, zero_division=0)),
        "average_precision": float(average_precision_score(y, prob)) if len(np.unique(y)) == 2 else None,
        "brier": float(brier_score_loss(y, prob)),
        "log_loss": float(log_loss(y, prob, labels=[0, 1])),
        "matriz_confusao": confusion_matrix(y, previsto, labels=[0, 1]).tolist(),
    }


def logito(prob):
    p = np.clip(np.asarray(prob), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def probabilidades(modelo, calibrador, dados):
    prob = modelo.predict_proba(caracteristicas(dados))[:, 1]
    return calibrador.predict_proba(logito(prob))[:, 1] if calibrador is not None else prob


def particionar(dados):
    elegiveis = dados[dados["fully_observed"]].copy()
    ano = elegiveis["period_start"].dt.year
    treino_local = elegiveis["spatial_group"].eq("train")
    partes = {
        "treino": elegiveis[treino_local & ano.between(2018, 2021)].copy(),
        "calibracao": elegiveis[treino_local & ano.eq(2022)].copy(),
        "decisao": elegiveis[treino_local & ano.eq(2023)].copy(),
        "teste_local": elegiveis[treino_local & ano.between(2024, 2025)].copy(),
        "teste_espacial": elegiveis[~treino_local & ano.between(2024, 2025)].copy(),
    }
    for nome, parte in partes.items():
        contagens = parte["event"].value_counts()
        if len(contagens) != 2 or contagens.min() < 2:
            raise ValueError(f"Particao {nome} exige pelo menos dois exemplos de cada classe: {contagens.to_dict()}.")
    return partes


def selecionar(treino):
    pontuacoes = {}
    for nome, modelo in candidatos().items():
        dobras = []
        for ano in [2019, 2020, 2021]:
            ajuste = treino[treino["period_start"].dt.year < ano]
            validacao = treino[treino["period_start"].dt.year == ano]
            if ajuste["event"].nunique() != 2 or validacao["event"].nunique() != 2:
                raise ValueError(f"Classes insuficientes na dobra temporal de {ano}.")
            modelo.fit(caracteristicas(ajuste), ajuste["event"].astype(int))
            prob = modelo.predict_proba(caracteristicas(validacao))[:, 1]
            dobras.append({
                "ano": ano, "treino": len(ajuste), "validacao": len(validacao),
                "positivos": int(validacao["event"].sum()),
                "brier": float(brier_score_loss(validacao["event"].astype(int), prob)),
            })
        pontuacoes[nome] = {"brier_medio": float(np.mean([x["brier"] for x in dobras])), "dobras": dobras}
    nome = min(pontuacoes, key=lambda n: pontuacoes[n]["brier_medio"])
    modelo = candidatos()[nome]
    modelo.fit(caracteristicas(treino), treino["event"].astype(int))
    return nome, modelo, pontuacoes


def escolher_limiar(y, prob):
    precisao, revocacao, limiares = precision_recall_curve(y, prob)
    soma = precisao[:-1] + revocacao[:-1]
    f1 = np.divide(2 * precisao[:-1] * revocacao[:-1], soma, out=np.zeros_like(soma), where=soma != 0)
    return float(limiares[np.argmax(f1)])


def referencia_sazonal(treino, dados):
    media = float(treino["event"].mean())
    grupo = treino.groupby(treino["period_start"].dt.month)["event"].agg(["sum", "count"])
    mensal = (grupo["sum"] + 10 * media) / (grupo["count"] + 10)
    return dados["period_start"].dt.month.map(mensal).fillna(media).to_numpy(dtype=float)


def treinar(caminho, pasta):
    caminho, pasta = Path(caminho), Path(pasta)
    dados = ler(caminho)
    partes = particionar(dados)
    treino, calibracao, decisao = (partes[x] for x in ["treino", "calibracao", "decisao"])
    nome, modelo, dobras = selecionar(treino)
    p_cal = probabilidades(modelo, None, calibracao)
    calibrador = LogisticRegression(max_iter=1000, random_state=42).fit(logito(p_cal), calibracao["event"].astype(int))
    p_dec = probabilidades(modelo, None, decisao)
    p_dec_cal = calibrador.predict_proba(logito(p_dec))[:, 1]
    brier_bruto = float(brier_score_loss(decisao["event"].astype(int), p_dec))
    brier_cal = float(brier_score_loss(decisao["event"].astype(int), p_dec_cal))
    if brier_cal >= brier_bruto:
        calibrador = None
    p_decisao = p_dec_cal if calibrador is not None else p_dec
    limiar = escolher_limiar(decisao["event"].astype(int), p_decisao)
    teste = pd.concat([partes["teste_local"], partes["teste_espacial"]]).sort_values(["period_start", "region_id"]).copy()
    teste["probabilidade_condicional"] = probabilidades(modelo, calibrador, teste)
    teste["classe"] = (teste["probabilidade_condicional"] >= limiar).astype(int)
    baseline = candidatos()["frequencia"]
    baseline.fit(caracteristicas(treino), treino["event"].astype(int))
    p_base = baseline.predict_proba(caracteristicas(teste))[:, 1]
    p_sazonal = referencia_sazonal(treino, teste)
    teste_sem_historico = teste.copy()
    teste_sem_historico[["past_viirs_mask_event", "past_viirs_log_records", "past_modis_event"]] = np.nan
    p_sem_historico = probabilidades(modelo, calibrador, teste_sem_historico)
    avaliacoes = {}
    for grupo, selecao in {
        "geral": pd.Series(True, index=teste.index),
        "areas_ajuste": teste["spatial_group"].eq("train"),
        "areas_reservadas": teste["spatial_group"].eq("holdout"),
    }.items():
        parte = teste[selecao]
        avaliacoes[grupo] = metricas(parte["event"], parte["probabilidade_condicional"], limiar)
    relatorio = {
        "criado_em_utc": datetime.now(timezone.utc).isoformat(),
        "modelo": nome, "sha256_dados": sha256(caminho), "produto": "VJ114A1.002",
        "objetivo": "Deteccao em pelo menos um pixel do recorte no dia UTC seguinte, condicionada a mascara integralmente classificada.",
        "escopo": "Retrospectivo e exploratorio; areas selecionadas, datas nao continuas e fontes possivelmente revisadas.",
        "dados": {
            "linhas": len(dados), "mascaras_incompletas": int((~dados["fully_observed"]).sum()),
            "incompletas_com_foco": int(((~dados["fully_observed"]) & dados["fire_pixels"].gt(0)).sum()),
        },
        "particoes": {
            n: {"linhas": len(p), "positivos": int(p["event"].sum()), "inicio": p["period_start"].min().date().isoformat(),
                "fim": p["period_start"].max().date().isoformat(), "areas": sorted(p["region_name"].unique().tolist())}
            for n, p in partes.items()
        },
        "selecao": {"metrica": "menor Brier medio nas dobras temporais do treino", "candidatos": dobras},
        "calibracao": {"usada": calibrador is not None, "brier_bruto_decisao": brier_bruto,
                       "brier_calibrado_decisao": brier_cal, "positivos_calibracao": int(calibracao["event"].sum())},
        "limiar_f1_decisao": limiar, "metricas_teste": avaliacoes,
        "baseline_frequencia_teste_limiar_0_5": metricas(teste["event"], p_base, 0.5),
        "baseline_sazonal_teste_limiar_0_5": metricas(teste["event"], p_sazonal, 0.5),
        "teste_sem_historico_satelite": metricas(teste["event"], p_sem_historico, limiar),
        "bibliotecas": {x: version(x) for x in ["numpy", "pandas", "scikit-learn", "joblib"]},
        "observacoes": [
            "As areas foram escolhidas por disponibilidade e atividade de focos, nao por amostragem representativa.",
            "A mascara completa futura nao e conhecida no instante da previsao; a probabilidade nao e incondicional.",
            "Os mesmos dados historicos foram examinados em experimento anterior; o teste e exploratorio, nao prospectivo.",
            "O periodo de calibracao e decisao tem poucos positivos; o limiar e a calibracao sao instaveis.",
            "FIRMS e POWER retrospectivos podem ter sido revisados; disponibilidade operacional passada nao foi comprovada.",
            "O teste sem historico de satelite foi acrescentado depois da primeira avaliacao e e apenas uma analise de sensibilidade.",
        ],
    }
    artefato = {
        "versao": 1, "modelo": modelo, "calibrador": calibrador, "limiar": limiar,
        "caracteristicas": CARACTERISTICAS, "sha256_dados": sha256(caminho),
        "avaliacao_sem_historico": relatorio["teste_sem_historico_satelite"],
        "referencia_sazonal": relatorio["baseline_sazonal_teste_limiar_0_5"],
        "regioes": dados[["region_id", "region_name", "latitude", "longitude", "spatial_group"]].drop_duplicates().to_dict("records"),
    }
    pasta.mkdir(parents=True, exist_ok=True)
    joblib.dump(artefato, pasta / "modelo_viirs.joblib")
    (pasta / "relatorio_viirs.json").write_text(json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")
    teste[["region_id", "region_name", "spatial_group", "period_start", "event", "probabilidade_condicional", "classe"]].to_csv(pasta / "previsoes_teste_viirs.csv", index=False)
    return relatorio


def main():
    parser = argparse.ArgumentParser(description="Treina e avalia o piloto VIIRS regional retrospectivo.")
    parser.add_argument("--dados", default=str(BASE))
    parser.add_argument("--saida", default=str(ARTEFATOS))
    args = parser.parse_args()
    relatorio = treinar(args.dados, args.saida)
    print(json.dumps({"modelo": relatorio["modelo"], "limiar": relatorio["limiar_f1_decisao"],
                      "teste": relatorio["metricas_teste"], "baseline": relatorio["baseline_frequencia_teste_limiar_0_5"],
                      "baseline_sazonal": relatorio["baseline_sazonal_teste_limiar_0_5"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
