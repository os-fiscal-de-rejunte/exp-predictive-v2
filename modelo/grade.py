import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import box, shape
from shapely.validation import make_valid

from modelo.config import RAIZ

LIMITES = RAIZ / "dados" / "natural_earth_50m_paises.geojson"
SAIDA = RAIZ / "dados" / "evolucao" / "grade_matopiba.csv"


def criar_grade(limites=LIMITES, oeste=-47.0, sul=-12.0, leste=-43.0, norte=-8.0, passo=0.25):
    if not (-180 <= oeste < leste <= 180 and -90 <= sul < norte <= 90 and passo > 0):
        raise ValueError("Area ou resolucao invalidas.")
    if abs(round((leste - oeste) / passo) * passo - (leste - oeste)) > 1e-8 or abs(round((norte - sul) / passo) * passo - (norte - sul)) > 1e-8:
        raise ValueError("Limites devem ser multiplos da resolucao.")
    arquivo = Path(limites)
    conteudo = arquivo.read_bytes()
    dados = json.loads(conteudo)
    paises = [f for f in dados["features"] if f.get("properties", {}).get("ADMIN") == "Brazil"]
    if len(paises) != 1:
        raise ValueError("Limite brasileiro ausente ou ambiguo.")
    brasil = make_valid(shape(paises[0]["geometry"]))
    origem = hashlib.sha256(conteudo).hexdigest()
    registros = []
    for i in range(round((norte - sul) / passo)):
        y0 = sul + i * passo
        for j in range(round((leste - oeste) / passo)):
            x0 = oeste + j * passo
            quadrado = box(x0, y0, x0 + passo, y0 + passo)
            intersecao = quadrado.intersection(brasil)
            if intersecao.is_empty or intersecao.area <= 0:
                continue
            fracao = intersecao.area / quadrado.area
            area = 6371.007181 ** 2 * np.deg2rad(passo) * (np.sin(np.deg2rad(y0 + passo)) - np.sin(np.deg2rad(y0)))
            registros.append({
                "grid_id": f"matopiba_{i:02d}_{j:02d}", "pais": "Brazil", "linha": i, "coluna": j,
                "latitude": y0 + passo / 2, "longitude": x0 + passo / 2,
                "sul": y0, "norte": y0 + passo, "oeste": x0, "leste": x0 + passo,
                "area_celula_km2": float(area), "fracao_brasil_aprox": float(fracao),
                "geometria_brasil_wkt": intersecao.wkt, "limite_sha256": origem,
            })
    if not registros:
        raise ValueError("Nenhuma area terrestre brasileira intersecta o recorte.")
    return pd.DataFrame(registros)


def main():
    parser = argparse.ArgumentParser(description="Cria grade independente dos focos no recorte piloto.")
    parser.add_argument("--limites", type=Path, default=LIMITES)
    parser.add_argument("--saida", type=Path, default=SAIDA)
    args = parser.parse_args()
    dados = criar_grade(args.limites)
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    dados.to_csv(args.saida, index=False, lineterminator="\r\n")
    print(f"Celulas terrestres no recorte: {len(dados)}; arquivo: {args.saida}")


if __name__ == "__main__":
    main()
