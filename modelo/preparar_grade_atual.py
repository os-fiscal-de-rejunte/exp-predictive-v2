import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from modelo.caracteristicas_grade import CLIMA, montar
from modelo.cobertura_solo import adicionar as adicionar_cobertura_solo
from modelo.coletar_mascaras import sha256
from modelo.config import RAIZ
from modelo.grade import SAIDA as GRADE

PASTA = RAIZ / "dados" / "evolucao_atual"


def estender_clima(clima, alvo):
    dados = clima.copy()
    dados["data_utc"] = pd.to_datetime(dados["data_utc"], utc=True, errors="raise")
    limite = pd.Timestamp(alvo, tz="UTC") - pd.Timedelta(days=3)
    if dados["data_utc"].max() > limite:
        raise ValueError("A consulta climatica inclui dados posteriores a D-3.")
    adicionais = []
    for (lat, lon), grupo in dados.groupby(["lat_power", "lon_power"], sort=False):
        ultima = grupo["data_utc"].max()
        for dia in pd.date_range(ultima + pd.Timedelta(days=1), pd.Timestamp(alvo, tz="UTC"), freq="D"):
            adicionais.append({"lat_power": lat, "lon_power": lon, "data_utc": dia, **{c: np.nan for c in CLIMA}})
    return pd.concat([dados, pd.DataFrame(adicionais)], ignore_index=True) if adicionais else dados


def preparar(mascaras, clima, firms, vegetacao, relevo, grade=GRADE, saida=None, cobertura_solo=None):
    instante = datetime.now(timezone.utc)
    alvo = instante.date() + timedelta(days=1)
    caminhos = [Path(x) for x in [mascaras, clima, firms, vegetacao, relevo, grade]]
    mascaras, clima, firms, vegetacao, relevo, grade = caminhos
    resumo_firms = firms.with_name("manifesto_" + firms.stem + ".json")
    metadados_firms = json.loads(resumo_firms.read_text(encoding="utf-8"))
    if metadados_firms.get("sha256_arquivo") != sha256(firms):
        raise ValueError("Arquivo FIRMS recente diverge do manifesto.")
    if set(metadados_firms.get("fontes", [])) != {"VIIRS_NOAA20_NRT", "MODIS_NRT"}:
        raise ValueError("A consulta atual exige as duas fontes NRT especificadas.")
    if pd.Timestamp(metadados_firms["inicio"]).date() > alvo - timedelta(days=8) or pd.Timestamp(metadados_firms["fim"]).date() < alvo - timedelta(days=2):
        raise ValueError("A janela recente FIRMS nao cobre os dias necessarios.")
    tempos_firms = [pd.Timestamp(r["cache_gravado_utc"]) for r in metadados_firms.get("respostas", [])]
    if not tempos_firms or any(pd.Timestamp(instante) - t > pd.Timedelta(hours=12) or t > pd.Timestamp(instante) for t in tempos_firms):
        raise ValueError("As respostas FIRMS recentes estao desatualizadas ou sem horario verificavel.")
    resumo_clima = clima.with_name("manifesto_" + clima.stem + ".json")
    metadados_clima = json.loads(resumo_clima.read_text(encoding="utf-8"))
    if metadados_clima.get("sha256_arquivo") != sha256(clima) or metadados_clima.get("fim") != (alvo - timedelta(days=3)).isoformat():
        raise ValueError("Arquivo climatico recente diverge do manifesto ou da janela necessaria.")
    tempos_clima = [pd.Timestamp(r["cache_gravado_utc"]) for r in metadados_clima.get("respostas", [])]
    if not tempos_clima or any(pd.Timestamp(instante) - t > pd.Timedelta(hours=12) or t > pd.Timestamp(instante) for t in tempos_clima):
        raise ValueError("As respostas climaticas recentes estao desatualizadas ou sem horario verificavel.")
    masks = pd.read_parquet(mascaras)
    masks["data_utc"] = pd.to_datetime(masks["data_utc"], utc=True, errors="raise")
    masks["producao_utc"] = pd.to_datetime(masks["producao_utc"], utc=True, errors="coerce")
    alvo_timestamp = pd.Timestamp(alvo, tz="UTC")
    linhas_alvo = masks[masks["data_utc"].eq(alvo_timestamp)]
    if len(linhas_alvo) != 256 or linhas_alvo["alvo_condicional"].notna().any() or linhas_alvo["mascara_completa"].any():
        raise ValueError("A base atual deve conter 256 celulas com alvo futuro desconhecido.")
    masks.loc[masks["producao_utc"] > pd.Timestamp(instante), "producao_utc"] = pd.NaT
    clima_base = estender_clima(pd.read_parquet(clima), alvo)
    detecoes = pd.read_parquet(firms)
    if not detecoes.empty:
        momentos = pd.to_datetime(detecoes["instante_utc"], utc=True, errors="raise")
        if momentos.max() > pd.Timestamp(instante):
            raise ValueError("Historico FIRMS contem deteccoes posteriores a coleta.")
    veg = pd.read_parquet(vegetacao)
    veg["disponivel_utc"] = pd.to_datetime(veg["disponivel_utc"], utc=True, errors="raise")
    veg = veg[veg["disponivel_utc"] <= pd.Timestamp(instante)].copy()
    mascaras_disponiveis = masks[masks["pixels_cobertos"].gt(0) & masks["producao_utc"].notna()]
    ultima_mascara = mascaras_disponiveis["data_utc"].max()
    vegetacao_valida = veg[veg["ndvi"].notna()]
    ultimo_composto = pd.to_datetime(vegetacao_valida["composto_fim"], errors="coerce").max()
    dados = montar(masks, clima_base, pd.read_csv(grade), detecoes, veg, pd.read_parquet(relevo))
    cobertura_solo = Path(cobertura_solo) if cobertura_solo else RAIZ / "dados" / "evolucao" / "cobertura_solo_grade.parquet"
    metadados_solo, solo = None, None
    if cobertura_solo.exists():
        metadados_solo = json.loads((cobertura_solo.parent / "resumo_cobertura_solo.json").read_text(encoding="utf-8"))
        if metadados_solo.get("sha256_arquivo") != sha256(cobertura_solo) or metadados_solo.get("sha256_grade") != sha256(grade):
            raise ValueError("Cobertura do solo divergiu do manifesto ou da grade.")
        solo = pd.read_parquet(cobertura_solo)
        solo["solo_disponivel_utc"] = pd.to_datetime(solo["solo_disponivel_utc"], utc=True, errors="raise")
        solo = solo[solo["solo_disponivel_utc"] <= pd.Timestamp(instante)].copy()
        if not solo.empty:
            dados = adicionar_cobertura_solo(dados, solo)
    if datetime.now(timezone.utc).date() != instante.date():
        raise RuntimeError("A data UTC mudou durante a preparacao; refaca a consulta.")
    saida = Path(saida) if saida else PASTA / f"caracteristicas_atuais_{alvo:%Y%m%d}.parquet"
    saida.parent.mkdir(parents=True, exist_ok=True)
    dados.to_parquet(saida, index=False)
    resumo = {"alvo_utc": alvo.isoformat(), "preparado_em_utc": datetime.now(timezone.utc).isoformat(),
              "limite_temporal_utc": instante.isoformat(), "celulas_alvo": len(linhas_alvo),
              "fontes": {"mascaras": sha256(mascaras), "clima": sha256(clima), "firms": sha256(firms),
                         "vegetacao": sha256(vegetacao), "relevo": sha256(relevo), "grade": sha256(grade),
                         "cobertura_solo": sha256(cobertura_solo) if metadados_solo else None},
              "fontes_firms": metadados_firms["fontes"], "sha256_saida": sha256(saida),
              "disponibilidade_fontes": {
                  "firms": {"inicio_utc": metadados_firms["inicio"], "fim_utc": metadados_firms["fim"],
                            "coleta_mais_antiga_utc": min(tempos_firms).isoformat(),
                            "coleta_mais_recente_utc": max(tempos_firms).isoformat()},
                  "clima": {"inicio_utc": metadados_clima["inicio"], "fim_utc": metadados_clima["fim"],
                            "coleta_mais_antiga_utc": min(tempos_clima).isoformat(),
                            "coleta_mais_recente_utc": max(tempos_clima).isoformat()},
                  "mascaras": {"ultima_data_com_arquivo_utc": None if pd.isna(ultima_mascara) else ultima_mascara.date().isoformat()},
                  "vegetacao": {"fim_ultimo_composto_valido_utc": None if pd.isna(ultimo_composto) else ultimo_composto.date().isoformat()},
                  "cobertura_solo": {"ultimo_ano": None if solo is None or solo.empty else int(solo["solo_ano"].max()),
                                     "produto": "MCD12Q1.061", "uso": "contexto, nao usado no modelo principal"},
              },
              "limite": "Consulta experimental; disponibilidade da mascara futura e cobertura de passagem sao desconhecidas."}
    saida.with_suffix(".json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    return resumo


def main():
    parser = argparse.ArgumentParser(description="Prepara entradas recentes para prever a grade regional.")
    parser.add_argument("--mascaras", type=Path, required=True)
    parser.add_argument("--clima", type=Path, required=True)
    parser.add_argument("--firms", type=Path, required=True)
    parser.add_argument("--vegetacao", type=Path, required=True)
    parser.add_argument("--relevo", type=Path, required=True)
    parser.add_argument("--grade", type=Path, default=GRADE)
    parser.add_argument("--saida", type=Path)
    parser.add_argument("--cobertura-solo", type=Path)
    args = parser.parse_args()
    print(json.dumps(preparar(args.mascaras, args.clima, args.firms, args.vegetacao, args.relevo, args.grade, args.saida, args.cobertura_solo), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
