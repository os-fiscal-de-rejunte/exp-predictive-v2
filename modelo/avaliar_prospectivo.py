import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, precision_score, recall_score

from modelo.coletar_mascaras import TILES, coletar, sha256
from modelo.grade import SAIDA as GRADE
from modelo.mascaras import processar
from modelo.preparar_grade_atual import PASTA as PASTA_ATUAL
from modelo.registro_prospectivo import PASTA, carregar


def selecionar_registros(pasta=PASTA):
    por_dia = {}
    for caminho in sorted(Path(pasta).glob("*/*.json")):
        registro = carregar(caminho)
        dia = registro["inicio_previsto_utc"][:10]
        if dia not in por_dia or registro["emitido_em_utc"] > por_dia[dia]["registro"]["emitido_em_utc"]:
            por_dia[dia] = {"registro": registro, "arquivo": str(caminho), "sha256": sha256(caminho)}
    return por_dia


def obter_mascaras(registros, pasta=PASTA_ATUAL, agora=None):
    agora = agora or datetime.now(timezone.utc)
    dias = sorted(d for d in registros if pd.Timestamp(d, tz="UTC").date() < agora.date())
    if not dias:
        return None
    inicio, fim = dias[0], dias[-1]
    resultado = coletar(inicio, fim, pasta)
    if resultado["falhas"]:
        raise RuntimeError("Falha na coleta das observacoes prospectivas.")
    manifesto = json.loads((Path(pasta) / "manifesto_viirs.json").read_text(encoding="utf-8"))
    tiles = {x["tile"] for x in manifesto["arquivos"].values() if inicio <= x["data"] <= fim}
    if not set(TILES).issubset(tiles):
        return None
    processar(inicio, fim, pasta)
    return Path(pasta) / f"mascaras_{inicio.replace('-', '')}_{fim.replace('-', '')}.parquet"


def ler_mascaras(caminho):
    caminho = Path(caminho)
    resumo = caminho.with_name("resumo_" + caminho.stem + ".json")
    metadados = json.loads(resumo.read_text(encoding="utf-8"))
    if metadados.get("sha256_resultado") != sha256(caminho) or metadados.get("sha256_grade") != sha256(GRADE):
        raise ValueError("Mascaras prospectivas divergiram do manifesto ou da grade.")
    dados = pd.read_parquet(caminho)
    dados["data_utc"] = pd.to_datetime(dados["data_utc"], utc=True, errors="raise")
    if dados.duplicated(["grid_id", "data_utc"]).any():
        raise ValueError("Observacoes prospectivas duplicadas.")
    completo = dados["mascara_completa"]
    if completo.isna().any() or not completo.isin([True, False]).all():
        raise ValueError("Estado da mascara prospectiva invalido.")
    if dados.loc[~completo, "alvo_condicional"].notna().any() or not dados.loc[completo, "alvo_condicional"].isin([0, 1]).all():
        raise ValueError("Rotulos prospectivos inconsistentes com a mascara.")
    return dados


def avaliar(registros, arquivo_mascaras=None, agora=None):
    agora = agora or datetime.now(timezone.utc)
    mascaras = ler_mascaras(arquivo_mascaras) if arquivo_mascaras else None
    resultados, linhas = [], []
    for dia, item in sorted(registros.items()):
        registro = item["registro"]
        inicio = pd.Timestamp(registro["inicio_previsto_utc"])
        fim = pd.Timestamp(registro["fim_previsto_utc"])
        if pd.Timestamp(registro["emitido_em_utc"]) >= inicio or pd.Timestamp(agora) < fim:
            resultados.append({"data_utc": dia, "status": "aguardando_fim_do_dia", "registro_sha256": item["sha256"]})
            continue
        obs = mascaras[mascaras["data_utc"].eq(inicio)] if mascaras is not None else pd.DataFrame()
        if obs.empty:
            resultados.append({"data_utc": dia, "status": "aguardando_mascaras", "registro_sha256": item["sha256"]})
            continue
        previsoes = pd.DataFrame(registro["celulas"])
        unido = previsoes.merge(obs[["grid_id", "mascara_completa", "alvo_condicional", "evidencia_foco"]],
                                left_on="celula_id", right_on="grid_id", how="left", validate="one_to_one")
        completos = unido["mascara_completa"].eq(True)
        com_previsao = unido["probabilidade_condicional"].notna()
        selecionados = unido[completos & com_previsao].copy()
        selecionados["sha256_modelo"] = registro["sha256_modelo"]
        selecionados["data_utc"] = dia
        linhas.append(selecionados)
        resultados.append({"data_utc": dia, "status": "observacoes_parciais" if not completos.all() else "observacoes_completas",
                           "celulas": len(unido), "observacoes_completas": int(completos.sum()),
                           "observacoes_desconhecidas": int((~completos).sum()),
                           "focos_em_mascara_incompleta": int((~completos & unido["evidencia_foco"].fillna(False)).sum()),
                           "comparadas": len(selecionados), "registro_sha256": item["sha256"]})
    grupos = []
    if linhas:
        tabela = pd.concat(linhas, ignore_index=True)
        for (modelo, modo), grupo in tabela.groupby(["sha256_modelo", "modo_modelo"]):
            if grupo.empty:
                continue
            y = grupo["alvo_condicional"].astype(int).to_numpy()
            p = grupo["probabilidade_condicional"].to_numpy(dtype=float)
            classes = grupo["classe_detectada"].to_numpy(dtype=int)
            positivos = int(y.sum())
            grupos.append({"sha256_modelo": modelo, "modo_modelo": modo,
                           "status": "avaliacao_exploratoria" if positivos >= 50 else "positivos_insuficientes",
                           "linhas": len(y), "positivos": positivos,
                           "acuracia": float(accuracy_score(y, classes)),
                           "precisao": float(precision_score(y, classes, zero_division=0)),
                           "revocacao": float(recall_score(y, classes, zero_division=0)),
                           "average_precision": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None,
                           "brier": float(brier_score_loss(y, p)),
                           "matriz_confusao": confusion_matrix(y, classes, labels=[0, 1]).tolist()})
    return {"gerado_em_utc": agora.isoformat(), "criterio": "Ultima emissao registrada antes do dia UTC, por data.",
            "dias": resultados, "grupos": grupos,
            "limites": "Teste observacional local: SHA-256 detecta alteracoes, mas nao comprova horario por terceiro. "
                       "Resultados condicionados a mascaras completas; menos de 50 positivos e inconclusivo. "
                       "Reamostragem espacial e temporal e teste independente adicional permanecem necessarios."}


def main():
    parser = argparse.ArgumentParser(description="Compara previsoes prospectivas registradas com mascaras VIIRS.")
    parser.add_argument("--registros", type=Path, default=PASTA)
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument("--mascaras", type=Path)
    grupo.add_argument("--atualizar-observacoes", action="store_true")
    parser.add_argument("--saida", type=Path, default=PASTA.parent / "avaliacao_prospectiva.json")
    args = parser.parse_args()
    registros = selecionar_registros(args.registros)
    mascaras = obter_mascaras(registros) if args.atualizar_observacoes else args.mascaras
    resultado = avaliar(registros, mascaras)
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    args.saida.write_text(json.dumps(resultado, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(resultado, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
