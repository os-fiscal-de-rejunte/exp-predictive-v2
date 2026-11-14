import argparse
from datetime import datetime, timezone
import json

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from modelo.config import ALVO, ARTEFATO, CATEGORICAS, ENTRADAS, NUMERICAS, RELATORIO
from modelo.dados import caracteristicas, ler_dados, montar_base


def candidatos():
    def criar(classificador):
        numerico = Pipeline([("imputar", SimpleImputer(strategy="median", keep_empty_features=True)), ("escalar", StandardScaler())])
        categorico = Pipeline([("imputar", SimpleImputer(strategy="constant", fill_value="__AUSENTE__")), ("codificar", OneHotEncoder(handle_unknown="ignore"))])
        preparo = ColumnTransformer([("numericas", numerico, NUMERICAS), ("categoricas", categorico, CATEGORICAS)])
        return Pipeline([("preparo", preparo), ("modelo", classificador)])
    return {
        "logistica": criar(LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)),
        "floresta": criar(RandomForestClassifier(n_estimators=250, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1)),
    }


def dividir(base):
    datas = np.sort(base["acq_date"].unique())
    if len(datas) < 16:
        raise ValueError(f"Sao necessarias pelo menos 16 datas validas; encontradas: {len(datas)}.")
    cortes = [int(len(datas) * p) for p in (0.6, 0.75, 0.875)]
    grupos = np.split(datas, cortes)
    partes = [base[base["acq_date"].isin(grupo)].copy() for grupo in grupos]
    for nome, parte in zip(("treino", "calibracao", "limiar", "teste"), partes):
        contagens = parte[ALVO].value_counts()
        if len(contagens) != 2 or contagens.min() < 2:
            raise ValueError(f"A particao {nome} precisa de ao menos dois exemplos de cada classe: {contagens.to_dict()}")
    return partes


def escolher_modelo(treino):
    datas = np.sort(treino["acq_date"].unique())
    if len(datas) < 8:
        raise ValueError("O treino precisa conter ao menos oito datas para validacao temporal.")
    resultados = {}
    for nome, modelo in candidatos().items():
        pontuacoes = []
        for indices_treino, indices_validacao in TimeSeriesSplit(n_splits=3).split(datas):
            ajuste = treino[treino["acq_date"].isin(datas[indices_treino])]
            validacao = treino[treino["acq_date"].isin(datas[indices_validacao])]
            if ajuste[ALVO].nunique() < 2 or validacao[ALVO].nunique() < 2:
                continue
            modelo.fit(caracteristicas(ajuste), ajuste[ALVO])
            prob = modelo.predict_proba(caracteristicas(validacao))[:, 1]
            pontuacoes.append(float(average_precision_score(validacao[ALVO], prob)))
        if len(pontuacoes) >= 2:
            resultados[nome] = float(np.mean(pontuacoes))
    if not resultados:
        raise ValueError("Nao ha classes suficientes em pelo menos duas dobras temporais do treino.")
    nome = max(resultados, key=resultados.get)
    modelo = candidatos()[nome]
    modelo.fit(caracteristicas(treino), treino[ALVO])
    return nome, modelo, resultados


def logito(prob):
    p = np.clip(np.asarray(prob), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dados", nargs="+", required=True)
    parser.add_argument("--artefato", type=str, default=str(ARTEFATO))
    parser.add_argument("--relatorio", type=str, default=str(RELATORIO))
    parser.add_argument("--confirmar-disponibilidade-entradas", action="store_true")
    args = parser.parse_args()
    if not args.confirmar_disponibilidade_entradas:
        parser.error("A base generica nao comprova quando clima e deteccoes ficaram disponiveis. Verifique as fontes e use --confirmar-disponibilidade-entradas; para o piloto auditado use modelo.treinar_viirs.")
    from pathlib import Path
    base = montar_base(ler_dados(args.dados))
    treino, calibracao, limiar, teste = dividir(base)
    nome, modelo, resultados = escolher_modelo(treino)
    p_cal = modelo.predict_proba(caracteristicas(calibracao))[:, 1]
    calibrador = LogisticRegression(random_state=42).fit(logito(p_cal), calibracao[ALVO])
    p_limiar_bruto = modelo.predict_proba(caracteristicas(limiar))[:, 1]
    p_limiar_cal = calibrador.predict_proba(logito(p_limiar_bruto))[:, 1]
    usar_calibrador = bool(brier_score_loss(limiar[ALVO], p_limiar_cal) < brier_score_loss(limiar[ALVO], p_limiar_bruto))
    p_limiar = p_limiar_cal if usar_calibrador else p_limiar_bruto
    precisao, cobertura, limiares = precision_recall_curve(limiar[ALVO], p_limiar)
    soma = precisao[:-1] + cobertura[:-1]
    f1 = np.divide(2 * precisao[:-1] * cobertura[:-1], soma, out=np.zeros_like(soma), where=soma != 0)
    corte = float(limiares[np.argmax(f1)])
    prob_teste = modelo.predict_proba(caracteristicas(teste))[:, 1]
    if usar_calibrador:
        prob_teste = calibrador.predict_proba(logito(prob_teste))[:, 1]
    accuracy = float(accuracy_score(teste[ALVO], prob_teste >= corte))
    relatorio = {
        "criado_em_utc": datetime.now(timezone.utc).isoformat(), "modelo": nome,
        "accuracy_teste": accuracy, "average_precision_teste": float(average_precision_score(teste[ALVO], prob_teste)),
        "brier_teste": float(brier_score_loss(teste[ALVO], prob_teste)), "limiar": corte,
        "calibrado": usar_calibrador, "validacao_temporal_average_precision": resultados,
        "interpretacao": "Amostra seletiva: positivos podem ter mascara ausente e negativos exigem observacao valida; as probabilidades nao representam automaticamente todas as celulas.",
        "disponibilidade_confirmada_pelo_operador": True,
        "particoes": {n: {"linhas": len(p), "inicio": str(p.acq_date.min()), "fim": str(p.acq_date.max()), "positivos": int(p[ALVO].sum())} for n, p in zip(("treino", "calibracao", "limiar", "teste"), (treino, calibracao, limiar, teste))},
    }
    artefato = {"modelo": modelo, "calibrador": calibrador if usar_calibrador else None, "limiar": corte, "entradas": ENTRADAS, "versao": 1}
    destino, caminho_relatorio = Path(args.artefato), Path(args.relatorio)
    destino.parent.mkdir(parents=True, exist_ok=True)
    caminho_relatorio.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artefato, destino)
    caminho_relatorio.write_text(json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Accuracy no teste temporal: {accuracy:.4f}")


if __name__ == "__main__":
    main()
