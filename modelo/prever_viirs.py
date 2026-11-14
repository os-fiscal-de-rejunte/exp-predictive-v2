import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from modelo.treinar_viirs import probabilidades
from modelo.viirs_dados import ARTEFATOS, BASE, CARACTERISTICAS, ler, sha256


def prever(caminho, artefato_path, pais, data, regiao=None):
    if pais.strip().casefold() not in {"brasil", "brazil"}:
        raise ValueError("Este piloto so contem oito recortes selecionados na amostra do Brasil.")
    artefato = joblib.load(artefato_path)
    if artefato.get("versao") != 1 or artefato.get("caracteristicas") != CARACTERISTICAS:
        raise ValueError("Contrato de caracteristicas incompativel.")
    if sha256(caminho) != artefato.get("sha256_dados"):
        raise ValueError("O piloto retrospectivo exige a mesma base auditada usada no treinamento.")
    dados = ler(caminho)
    data = pd.to_datetime(data, utc=True, errors="raise")
    if data != data.normalize():
        raise ValueError("Informe uma data de referencia inteira no calendario UTC.")
    data_alvo = data + pd.Timedelta(days=1)
    selecao = dados["period_start"].eq(data_alvo)
    if regiao:
        selecao &= dados["region_name"].str.casefold().eq(regiao.strip().casefold())
    selecionados = dados[selecao].copy()
    if selecionados.empty:
        raise ValueError("Nao ha recortes historicos nessa data e regiao. Consulte as datas e regioes disponiveis na base.")
    probs = probabilidades(artefato["modelo"], artefato["calibrador"], selecionados)
    recortes = []
    for linha, prob in zip(selecionados.itertuples(index=False), probs):
        recortes.append({
            "regiao": linha.region_name, "latitude_centro": linha.latitude, "longitude_centro": linha.longitude,
            "grupo_espacial": "reservado" if linha.spatial_group == "holdout" else "ajuste",
            "probabilidade_condicional": float(prob), "classe": int(prob >= artefato["limiar"]),
        })
    return {
        "situacao": "experimental_retrospectivo", "pais_amostra": "Brasil",
        "data_referencia_utc": data.date().isoformat(), "data_prevista_utc": data_alvo.date().isoformat(),
        "produto": "VJ114A1.002, VIIRS NOAA-20, FireMask diario de 1 km",
        "interpretacao": "Probabilidade de detectar ao menos um pixel de foco no recorte, condicionada a classificacao valida de todos os pixels da mascara futura.",
        "probabilidade_incendio_incondicional": None,
        "aviso": "Dados historicos possivelmente revisados; oito areas selecionadas nao representam o pais nem permitem previsao operacional.",
        "limiar_classe": artefato["limiar"], "recortes": recortes,
    }


def main():
    parser = argparse.ArgumentParser(description="Consulta uma emissao historica do piloto VIIRS regional.")
    parser.add_argument("--dados", default=str(BASE))
    parser.add_argument("--artefato", default=str(ARTEFATOS / "modelo_viirs.joblib"))
    parser.add_argument("--pais", required=True)
    parser.add_argument("--data", required=True, help="Data de referencia UTC, um dia antes do alvo.")
    parser.add_argument("--regiao")
    parser.add_argument("--saida", type=Path)
    args = parser.parse_args()
    resultado = prever(args.dados, args.artefato, args.pais, args.data, args.regiao)
    texto = json.dumps(resultado, ensure_ascii=False, indent=2)
    if args.saida:
        args.saida.parent.mkdir(parents=True, exist_ok=True)
        args.saida.write_text(texto, encoding="utf-8")
    else:
        print(texto)


if __name__ == "__main__":
    main()
