import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, precision_score, recall_score

from modelo.coletar_mascaras import PASTA, sha256
from modelo.treinar_grade import SAIDA


def calibracao(y, prob):
    grupos = np.minimum((prob * 10).astype(int), 9)
    faixas = []
    for i in range(10):
        indice = grupos == i
        if not indice.any():
            continue
        faixas.append({"intervalo": [i / 10, (i + 1) / 10], "linhas": int(indice.sum()),
                       "positivos": int(y[indice].sum()), "probabilidade_media": float(prob[indice].mean()),
                       "frequencia_observada": float(y[indice].mean())})
    erro = sum(x["linhas"] * abs(x["probabilidade_media"] - x["frequencia_observada"]) for x in faixas) / len(y)
    return {"erro_absoluto_medio_por_faixa": float(erro), "faixas": faixas}


def auditar(previsoes, relatorio, mascaras):
    previsoes, relatorio, mascaras = map(Path, [previsoes, relatorio, mascaras])
    dados = pd.read_csv(previsoes)
    referencia = json.loads(relatorio.read_text(encoding="utf-8"))
    if dados.duplicated(["grid_id", "data_utc"]).any() or dados["coluna"].isin([10, 11]).any():
        raise ValueError("Duplicata ou area de afastamento presente no teste.")
    y = dados["alvo_condicional"].to_numpy(dtype=int)
    modos = {}
    for modo, registro in referencia["modos"].items():
        prob = dados["prob_" + modo].to_numpy(dtype=float)
        classes = dados["classe_" + modo].to_numpy(dtype=int)
        if not np.isfinite(prob).all() or ((prob < 0) | (prob > 1)).any():
            raise ValueError(f"Probabilidade invalida no modo {modo}.")
        metricas = {"linhas": len(y), "positivos": int(y.sum()), "acuracia": accuracy_score(y, classes),
                    "precisao": precision_score(y, classes, zero_division=0),
                    "revocacao": recall_score(y, classes, zero_division=0),
                    "average_precision": average_precision_score(y, prob), "brier": brier_score_loss(y, prob)}
        esperado = registro["teste"]["geral"]
        if not all(np.isclose(valor, esperado[nome], atol=1e-12) for nome, valor in metricas.items()):
            raise ValueError(f"Metricas divergiram no modo {modo}.")
        if confusion_matrix(y, classes, labels=[0, 1]).tolist() != esperado["matriz_confusao"]:
            raise ValueError(f"Matriz de confusao divergiu no modo {modo}.")
        if not np.array_equal(classes, (prob >= registro["limiar"]).astype(int)):
            raise ValueError(f"Classes e limiar divergiram no modo {modo}.")
        modos[modo] = calibracao(y, prob)
    obs = pd.read_parquet(mascaras)
    if obs.duplicated(["grid_id", "data_utc"]).any():
        raise ValueError("Celulas e datas duplicadas na base de mascaras.")
    resumo = json.loads(mascaras.with_name("resumo_" + mascaras.stem + ".json").read_text(encoding="utf-8"))
    if resumo["sha256_resultado"] != sha256(mascaras):
        raise ValueError("Base de mascaras divergiu do manifesto.")
    obs["mes"] = pd.to_datetime(obs["data_utc"], utc=True).dt.strftime("%Y-%m")
    cobertura = []
    for mes, grupo in obs.groupby("mes"):
        completos = grupo["mascara_completa"]
        cobertura.append({"mes": mes, "linhas": len(grupo), "completas": int(completos.sum()),
                          "frac_completas": float(completos.mean()),
                          "positivas_completas": int((completos & grupo["evidencia_foco"]).sum()),
                          "positivas_incompletas": int((~completos & grupo["evidencia_foco"]).sum())})
    return {"criado_em_utc": datetime.now(timezone.utc).isoformat(), "sha256_previsoes": sha256(previsoes),
            "sha256_relatorio": sha256(relatorio), "sha256_mascaras": sha256(mascaras),
            "status": "metricas_recalculadas_conferem", "calibracao_no_teste": modos,
            "cobertura_mensal": cobertura,
            "limites": "Teste retrospectivo ja explorado. Faixas com poucos casos sao instaveis; "
                       "a dependencia entre celulas e dias impede tratar as linhas como observacoes independentes."}


def main():
    parser = argparse.ArgumentParser(description="Audita metricas, calibracao e cobertura da grade regional.")
    parser.add_argument("--previsoes", type=Path, default=SAIDA / "previsoes_teste.csv")
    parser.add_argument("--relatorio", type=Path, default=SAIDA / "relatorio_grade.json")
    parser.add_argument("--mascaras", type=Path, default=PASTA / "mascaras_20240101_20251231.parquet")
    parser.add_argument("--saida", type=Path, default=SAIDA / "auditoria_grade.json")
    args = parser.parse_args()
    resultado = auditar(args.previsoes, args.relatorio, args.mascaras)
    args.saida.write_text(json.dumps(resultado, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": resultado["status"], "arquivo": str(args.saida),
                      "meses_cobertura": len(resultado["cobertura_mensal"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
