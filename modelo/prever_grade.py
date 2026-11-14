import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from modelo.coletar_mascaras import sha256
from modelo.config import RAIZ
from modelo.grade import SAIDA as GRADE
from modelo.treinar_grade import SAIDA as ARTEFATOS, probabilidade

SAIDA = RAIZ / "modelo" / "artefatos" / "evolucao_grade" / "previsao_atual.json"


def valor(numero):
    return None if pd.isna(numero) else float(numero)


def inteiro(numero):
    return None if numero is None or pd.isna(numero) else int(numero)


def prever(dados_path, modelo_path, relatorio_path, grade_path=GRADE):
    dados_path, modelo_path, relatorio_path, grade_path = map(Path, [dados_path, modelo_path, relatorio_path, grade_path])
    metadados = json.loads(dados_path.with_suffix(".json").read_text(encoding="utf-8"))
    if metadados.get("sha256_saida") != sha256(dados_path) or metadados.get("fontes", {}).get("grade") != sha256(grade_path):
        raise ValueError("Dados atuais ou grade divergiram do manifesto.")
    agora = datetime.now(timezone.utc)
    data_alvo = agora.date() + timedelta(days=1)
    instante = pd.Timestamp(metadados["preparado_em_utc"])
    if metadados.get("alvo_utc") != data_alvo.isoformat() or instante > pd.Timestamp(agora) or pd.Timestamp(agora) - instante > pd.Timedelta(hours=12):
        raise ValueError("A preparacao atual esta fora da janela valida; colete os dados novamente.")
    if set(metadados.get("fontes_firms", [])) != {"VIIRS_NOAA20_NRT", "MODIS_NRT"}:
        raise ValueError("A origem recente FIRMS nao foi comprovada.")
    artefato = joblib.load(modelo_path)
    relatorio = json.loads(relatorio_path.read_text(encoding="utf-8"))
    if artefato.get("versao") != 2 or artefato.get("sha256_dados") != relatorio.get("sha256_dados"):
        raise ValueError("Modelo e relatorio pertencem a treinamentos diferentes.")
    grade = pd.read_csv(grade_path).sort_values(["linha", "coluna"]).reset_index(drop=True)
    if len(grade) != 256 or grade["grid_id"].duplicated().any():
        raise ValueError("Grade piloto invalida.")
    dados = pd.read_parquet(dados_path)
    dados["data_utc"] = pd.to_datetime(dados["data_utc"], utc=True, errors="raise")
    alvo = dados[dados["data_utc"].eq(pd.Timestamp(data_alvo, tz="UTC"))].copy()
    if len(alvo) != 256 or alvo["grid_id"].duplicated().any() or set(alvo["grid_id"]) != set(grade["grid_id"]):
        raise ValueError("A consulta nao cobre exatamente as celulas da grade validada.")
    if alvo["alvo_condicional"].notna().any() or alvo["mascara_completa"].any():
        raise ValueError("O alvo futuro nao pode conter observacao real.")
    colunas_clima = ["clima_T2M", "clima_RH2M", "clima_PRECTOTCORR", "clima_WS10M", "clima_GWETROOT"]
    colunas_geo = ["vegetacao_ndvi", "vegetacao_evi", "altitude_media_m", "altitude_desvio_m"]
    tem_clima = alvo[colunas_clima].notna().all(axis=1)
    tem_geo = alvo[colunas_geo].notna().all(axis=1)
    tem_mascara = alvo["mascara_conhecidos_7d"].fillna(0).ge(3)
    alvo["modo_modelo"] = np.where(tem_mascara, "com_geografia", "firms_sem_mascara")
    alvo.loc[~tem_geo & tem_clima, "modo_modelo"] = "somente_clima"
    alvo.loc[~tem_clima, "modo_modelo"] = "dados_insuficientes"
    alvo["probabilidade_condicional"] = np.nan
    for modo in alvo["modo_modelo"].unique():
        if modo == "dados_insuficientes":
            continue
        if modo not in artefato["modos"] or modo not in relatorio["modos"]:
            raise ValueError(f"Modo nao treinado ou sem avaliacao: {modo}.")
        filtro = alvo["modo_modelo"].eq(modo)
        pacote = artefato["modos"][modo]
        alvo.loc[filtro, "probabilidade_condicional"] = probabilidade(pacote["modelo"], pacote["calibrador"], alvo.loc[filtro], pacote["colunas"])
    alvo = alvo.merge(grade[["grid_id", "sul", "norte", "oeste", "leste"]], on="grid_id", validate="one_to_one")
    regioes = []
    for item in alvo.sort_values("grid_id").itertuples(index=False):
        limiar = None if item.modo_modelo == "dados_insuficientes" else float(artefato["modos"][item.modo_modelo]["limiar"])
        classe = None if limiar is None else int(item.probabilidade_condicional >= limiar)
        regioes.append({
            "celula_id": item.grid_id, "latitude": item.latitude, "longitude": item.longitude,
            "limites": [item.oeste, item.sul, item.leste, item.norte],
            "status": "dados_insuficientes" if item.modo_modelo == "dados_insuficientes" else "experimental",
            "motivo": "Dados meteorologicos incompletos." if item.modo_modelo == "dados_insuficientes" else None,
            "modo_modelo": item.modo_modelo, "probabilidade_condicional": valor(item.probabilidade_condicional),
            "limiar_classificacao": limiar, "classe_detectada": classe,
            "intervalo_incerteza": None, "historico_mascara_dias_conhecidos": int(item.mascara_conhecidos_7d),
            "viirs_registros_7d": valor(item.viirs_registros_7d), "modis_registros_7d": valor(item.modis_registros_7d),
            "viirs_soma_frp_mw_7d": valor(item.viirs_frp_7d), "modis_soma_frp_mw_7d": valor(item.modis_frp_7d),
            "clima_historico": {"temperatura_media_c": valor(item.clima_T2M), "umidade_relativa_pct": valor(item.clima_RH2M),
                                "chuva_8_dias_mm": valor(item.clima_PRECTOTCORR), "vento_ms": valor(item.clima_WS10M),
                                "umidade_solo": valor(item.clima_GWETROOT)},
            "vegetacao": {"ndvi": valor(item.vegetacao_ndvi), "evi": valor(item.vegetacao_evi),
                          "idade_dias": valor(item.vegetacao_idade_dias)},
            "altitude_media_m": valor(item.altitude_media_m),
            "cobertura_solo": {"ano": inteiro(getattr(item, "solo_ano", None)),
                               "classe_dominante_igbp": inteiro(getattr(item, "solo_classe_dominante", None)),
                               "fracao_floresta": valor(getattr(item, "solo_floresta_frac", None)),
                               "fracao_savana_campo": valor(getattr(item, "solo_savana_campo_frac", None)),
                               "fracao_cultivo_mosaico": valor(getattr(item, "solo_cultivo_frac", None)),
                               "fracao_pixels_validos": valor(getattr(item, "solo_qualidade_frac", None))},
        })
    usados = sorted(set(x["modo_modelo"] for x in regioes) - {"dados_insuficientes"})
    return {"status": "experimental_nao_operacional", "emitido_em_utc": agora.isoformat(),
            "inicio_previsto_utc": data_alvo.isoformat() + "T00:00:00Z",
            "fim_previsto_utc": (data_alvo + timedelta(days=1)).isoformat() + "T00:00:00Z",
            "area": "Recorte piloto MATOPIBA, Brasil", "celulas": regioes,
            "produto_alvo": "VJ114A1.002 VIIRS NOAA-20, mascara diaria nominal de 1 km",
            "resolucao_grade_graus": 0.25,
            "interpretacao": "Probabilidade condicional de detectar ao menos um pixel de foco VIIRS em cada celula, caso toda a mascara futura seja classificada.",
            "probabilidade_incendio_incondicional": None, "probabilidade_nacional": None,
            "fontes": metadados["fontes"], "sha256_modelo": sha256(modelo_path),
            "disponibilidade_fontes": metadados.get("disponibilidade_fontes", {}),
            "avaliacoes_retrospectivas": {m: relatorio["modos"][m]["teste"] for m in usados},
            "aviso": "Dados de treino reprocessados e teste exploratorio em uma unica regiao; FIRMS NRT difere do historico SP. Nao usar como alerta operacional."}


def geojson(resultado):
    feicoes = []
    for celula in resultado["celulas"]:
        oeste, sul, leste, norte = celula["limites"]
        propriedades = {k: celula.get(k) for k in ["celula_id", "status", "motivo", "modo_modelo", "probabilidade_condicional", "limiar_classificacao", "classe_detectada"]}
        propriedades["cobertura_solo"] = celula.get("cobertura_solo")
        feicoes.append({"type": "Feature", "geometry": {"type": "Polygon",
                        "coordinates": [[[oeste, sul], [leste, sul], [leste, norte], [oeste, norte], [oeste, sul]]]},
                        "properties": propriedades})
    return {"type": "FeatureCollection", "features": feicoes,
            "interpretacao": resultado["interpretacao"], "status": resultado["status"],
            "emitido_em_utc": resultado.get("emitido_em_utc"),
            "inicio_previsto_utc": resultado.get("inicio_previsto_utc"),
            "fim_previsto_utc": resultado.get("fim_previsto_utc"),
            "disponibilidade_fontes": resultado.get("disponibilidade_fontes", {})}


def main():
    parser = argparse.ArgumentParser(description="Gera previsoes experimentais para a grade regional.")
    parser.add_argument("--dados", type=Path, required=True)
    parser.add_argument("--modelo", type=Path, default=ARTEFATOS / "modelo_grade.joblib")
    parser.add_argument("--relatorio", type=Path, default=ARTEFATOS / "relatorio_grade.json")
    parser.add_argument("--grade", type=Path, default=GRADE)
    parser.add_argument("--saida", type=Path, default=SAIDA)
    args = parser.parse_args()
    resultado = prever(args.dados, args.modelo, args.relatorio, args.grade)
    from modelo.registro_prospectivo import registrar
    registro = registrar(resultado)
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    args.saida.write_text(json.dumps(resultado, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    mapa = args.saida.with_suffix(".geojson")
    mapa.write_text(json.dumps(geojson(resultado), ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"arquivo": str(args.saida), "geojson": str(mapa), "registro_prospectivo": str(registro), "celulas": len(resultado["celulas"]),
                      "alvo_utc": resultado["inicio_previsto_utc"], "status": resultado["status"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
