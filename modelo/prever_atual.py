import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

from modelo.treinar_viirs import probabilidades
from modelo.historico_firms import consultar as consultar_historico
from modelo.config import RAIZ
from modelo.viirs_dados import ARTEFATOS, BASE, CARACTERISTICAS, sha256

URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
UNIDADES = {"T2M": "C", "RH2M": "%", "PRECTOTCORR": "mm/day", "WS10M": "m/s", "GWETROOT": "1"}


def extrair_clima(corpo, inicio, fim):
    header = corpo.get("header", {})
    if header.get("time_standard") != "UTC" or header.get("start") != inicio.strftime("%Y%m%d") or header.get("end") != fim.strftime("%Y%m%d"):
        raise ValueError("Resposta POWER com periodo ou fuso inesperado.")
    parametros = corpo.get("properties", {}).get("parameter", {})
    dias = pd.date_range(inicio, fim, freq="D").strftime("%Y%m%d").tolist()
    if len(dias) != 8:
        raise ValueError("A janela meteorologica deve conter oito dias.")
    resultado = {}
    for nome, unidade in UNIDADES.items():
        if corpo.get("parameters", {}).get(nome, {}).get("units") != unidade:
            raise ValueError(f"Unidade POWER inesperada: {nome}.")
        valores = parametros.get(nome, {})
        if any(dia not in valores or valores[dia] is None or valores[dia] == header.get("fill_value") for dia in dias):
            raise ValueError(f"Janela meteorologica incompleta: {nome}.")
        vetor = np.asarray([valores[dia] for dia in dias], dtype=float)
        if not np.isfinite(vetor).all():
            raise ValueError(f"Valor meteorologico invalido: {nome}.")
        resultado["weather_" + nome] = float(vetor.sum() if nome == "PRECTOTCORR" else vetor.mean())
    return resultado


def consultar_clima(latitude, longitude, inicio, fim):
    parametros = {
        "parameters": ",".join(UNIDADES), "community": "AG", "latitude": latitude, "longitude": longitude,
        "start": inicio.strftime("%Y%m%d"), "end": fim.strftime("%Y%m%d"),
        "format": "JSON", "time-standard": "UTC",
    }
    try:
        resposta = requests.get(URL, params=parametros, timeout=30)
        resposta.raise_for_status()
    except requests.RequestException as erro:
        codigo = getattr(getattr(erro, "response", None), "status_code", None)
        raise RuntimeError(f"NASA POWER indisponivel{f' (HTTP {codigo})' if codigo else ''}.") from None
    try:
        corpo = resposta.json()
        coordenadas = corpo.get("geometry", {}).get("coordinates", [])
        if len(coordenadas) < 2 or abs(float(coordenadas[0]) - longitude) > 0.001 or abs(float(coordenadas[1]) - latitude) > 0.001:
            raise ValueError("Coordenadas da resposta diferem da consulta.")
        valores = extrair_clima(corpo, inicio, fim)
    except (ValueError, KeyError, TypeError) as erro:
        raise RuntimeError(f"Resposta meteorologica inutilizavel: {erro}") from None
    return valores, hashlib.sha256(resposta.content).hexdigest()


def prever_atual(caminho, artefato_path, pais, regiao=None):
    if pais.strip().casefold() not in {"brasil", "brazil"}:
        raise ValueError("Este piloto so contem recortes selecionados na amostra do Brasil.")
    artefato = joblib.load(artefato_path)
    if artefato.get("versao") not in {1, 2} or artefato.get("caracteristicas") != CARACTERISTICAS:
        raise ValueError("Contrato de caracteristicas incompativel.")
    if sha256(caminho) != artefato.get("sha256_dados"):
        raise ValueError("A base local nao corresponde ao artefato treinado.")
    instante_inicio = datetime.now(timezone.utc)
    data_alvo = instante_inicio.date() + timedelta(days=1)
    inicio = data_alvo - timedelta(days=10)
    fim = data_alvo - timedelta(days=3)
    regioes = artefato["regioes"]
    if regiao:
        regioes = [r for r in regioes if r["region_name"].casefold() == regiao.strip().casefold()]
    if not regioes:
        raise ValueError("Regiao nao contemplada pelo piloto.")
    saidas = []
    for item in regioes:
        resultado = {"regiao": item["region_name"], "latitude_centro": item["latitude"],
                     "longitude_centro": item["longitude"], "grupo_espacial": item["spatial_group"]}
        try:
            clima, hash_resposta = consultar_clima(item["latitude"], item["longitude"], inicio, fim)
            historico = None
            modo = "legado_sem_satelite"
            contagem_viirs = np.nan
            if artefato["versao"] == 2:
                try:
                    historico = consultar_historico(item, data_alvo)
                except RuntimeError as erro:
                    historico = {"viirs": {"situacao": "indisponivel", "motivo": str(erro)},
                                 "modis": {"situacao": "indisponivel", "motivo": str(erro)}}
                viirs = historico.get("viirs", {})
                modo = "viirs_firms" if viirs.get("situacao") == "consultado" else "sem_satelite"
                if viirs.get("registros_positivos") is not None:
                    contagem_viirs = np.log1p(viirs["registros_positivos"])
                selecionado = artefato["modos"][modo]
            else:
                selecionado = artefato
            entrada = pd.DataFrame([{
                **item, **clima, "period_start": pd.Timestamp(data_alvo, tz="UTC"),
                "past_viirs_mask_event": np.nan, "past_viirs_log_records": contagem_viirs, "past_modis_event": np.nan,
            }])
            prob = probabilidades(selecionado["modelo"], selecionado["calibrador"], entrada)[0]
            resultado.update({"situacao": "estimativa_experimental", "probabilidade_condicional": float(prob),
                              "modo_modelo": modo, "historico_firms": historico,
                              "clima_historico": clima, "sha256_resposta_power": hash_resposta})
        except RuntimeError as erro:
            resultado.update({"situacao": "dados_indisponiveis", "probabilidade_condicional": None, "motivo": str(erro)})
        saidas.append(resultado)
    instante_fim = datetime.now(timezone.utc)
    if instante_fim.date() != instante_inicio.date():
        raise RuntimeError("A data UTC mudou durante a coleta; execute a consulta novamente.")
    return {
        "situacao": "experimental_nao_operacional", "emitido_em_utc": instante_fim.isoformat(),
        "pais_amostra": "Brasil", "data_prevista_utc": data_alvo.isoformat(),
        "sha256_modelo": sha256(artefato_path), "sha256_base_treinamento": artefato["sha256_dados"],
        "clima_de_utc": inicio.isoformat(), "clima_ate_utc": fim.isoformat(), "fonte_clima": URL,
        "produto_alvo": "VJ114A1.002, VIIRS NOAA-20, FireMask diario de 1 km",
        "interpretacao": "Probabilidade experimental de ao menos um pixel de foco no recorte, condicionada a classificacao valida de toda a mascara futura.",
        "probabilidade_incendio_incondicional": None,
        "historico_modis_e_viirs": ("Registros FIRMS recentes por recorte, quando disponiveis; ausencia de registros nao comprova ausencia de foco. MODIS e informado como contexto." if artefato["versao"] == 2 else "Indisponivel nesta consulta; o modelo usa imputacao e indicadores de ausencia."),
        "avaliacao_retrospectiva_sem_historico": artefato.get("avaliacao_sem_historico"),
        "referencia_sazonal_no_mesmo_teste": artefato.get("referencia_sazonal"),
        "avaliacao_por_modo": artefato.get("avaliacao"), "referencias_por_modo": artefato.get("referencias"),
        "aviso": "Avaliacao retrospectiva exploratoria; FIRMS NRT pode diferir dos dados historicos SP. A observacao futura e desconhecida. Nao use como alerta operacional.",
        "recortes": saidas,
    }


def main():
    parser = argparse.ArgumentParser(description="Consulta experimental para o proximo dia UTC com clima POWER recente.")
    parser.add_argument("--pais", required=True)
    parser.add_argument("--regiao")
    parser.add_argument("--dados", default=str(BASE))
    novo = RAIZ / "modelo" / "artefatos" / "evolucao_recortes" / "modelo_recortes.joblib"
    parser.add_argument("--artefato", default=str(novo if novo.exists() else ARTEFATOS / "modelo_viirs.joblib"))
    parser.add_argument("--saida", type=Path)
    args = parser.parse_args()
    resultado = prever_atual(args.dados, args.artefato, args.pais, args.regiao)
    texto = json.dumps(resultado, ensure_ascii=False, indent=2)
    if args.saida:
        args.saida.parent.mkdir(parents=True, exist_ok=True)
        args.saida.write_text(texto, encoding="utf-8")
    else:
        print(texto)


if __name__ == "__main__":
    main()
