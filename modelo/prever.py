import argparse
import json

import joblib

from modelo.config import ARTEFATO, ENTRADAS
from modelo.dados import caracteristicas, ler_dados
from modelo.treinar import logito


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dados", nargs="+", required=True)
    parser.add_argument("--pais", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--artefato", default=str(ARTEFATO))
    args = parser.parse_args()
    import pandas as pd
    artefato = joblib.load(args.artefato)
    if artefato.get("versao") != 1 or artefato.get("entradas") != ENTRADAS:
        raise ValueError("Contrato de caracteristicas incompativel.")
    dados = ler_dados(args.dados)
    data = pd.to_datetime(args.data, utc=True, errors="raise").normalize()
    filtro = dados["country"].str.casefold().eq(args.pais.strip().casefold()).fillna(False)
    filtro &= dados["acq_date"].eq(data)
    filtro &= dados["hotspot_count"].gt(0) | dados["has_valid_satellite_obs"].eq(True).fillna(False)
    selecionados = dados[filtro].copy()
    if selecionados.empty:
        raise ValueError("Nenhuma celula com deteccao ou observacao valida para o pais e data informados.")
    prob = artefato["modelo"].predict_proba(caracteristicas(selecionados))[:, 1]
    if artefato["calibrador"] is not None:
        prob = artefato["calibrador"].predict_proba(logito(prob))[:, 1]
    saidas = []
    for (_, linha), p in zip(selecionados.iterrows(), prob):
        saidas.append({"grid_id": linha.grid_id, "latitude": linha.grid_lat, "longitude": linha.grid_lon, "probabilidade_na_amostra_observavel": float(p), "classe": int(p >= artefato["limiar"])})
    print(json.dumps({"pais": args.pais, "data_referencia_utc": data.date().isoformat(), "data_prevista_utc": (data + pd.Timedelta(days=1)).date().isoformat(), "aviso": "Amostra seletiva: positivos podem ter mascara ausente e negativos exigem observacao valida; nao e chance incondicional de incendio.", "celulas": saidas}, ensure_ascii=False))


if __name__ == "__main__":
    main()
