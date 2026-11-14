import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd
from shapely.geometry import box, shape
from shapely.validation import make_valid

from modelo.config import RAIZ

PASTA = RAIZ / "dados" / "teste_suica"
LIMITES = RAIZ / "dados" / "natural_earth_50m_paises.geojson"
GRADE = PASTA / "grade_suica.csv"
AREA = "5.75,45.75,10.75,47.75"
TILES = {"h18v04"}


def criar_grade():
    conteudo = LIMITES.read_bytes()
    origem = json.loads(conteudo)
    paises = [x for x in origem["features"] if x.get("properties", {}).get("ADMIN") == "Switzerland"]
    if len(paises) != 1:
        raise ValueError("Limite da Suica ausente ou ambiguo.")
    pais = make_valid(shape(paises[0]["geometry"]))
    linhas = []
    for linha in range(8):
        for coluna in range(20):
            sul = 45.75 + linha * 0.25
            oeste = 5.75 + coluna * 0.25
            if pais.covers(box(oeste, sul, oeste + 0.25, sul + 0.25)):
                linhas.append({"grid_id": f"suica_{linha:02d}_{coluna:02d}", "pais": "Switzerland",
                               "linha": linha, "coluna": coluna, "sul": sul, "norte": sul + 0.25,
                               "oeste": oeste, "leste": oeste + 0.25,
                               "latitude": sul + 0.125, "longitude": oeste + 0.125})
    grade = pd.DataFrame(linhas)
    if grade.empty or grade["grid_id"].duplicated().any():
        raise ValueError("Grade suica vazia ou duplicada.")
    PASTA.mkdir(parents=True, exist_ok=True)
    grade.to_csv(GRADE, index=False, lineterminator="\r\n")
    resumo = {"pais": "Switzerland", "area_consultada": AREA, "celulas": len(grade),
              "resolucao_graus": 0.25, "criterio": "Celulas totalmente contidas no limite Natural Earth; regioes fronteiricas excluidas.",
              "sha256_limites": hashlib.sha256(conteudo).hexdigest(),
              "sha256_grade": hashlib.sha256(GRADE.read_bytes()).hexdigest(),
              "criado_em_utc": datetime.now(timezone.utc).isoformat()}
    (PASTA / "manifesto_grade.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    return resumo


def nomes_terreno():
    grade = pd.read_csv(GRADE)
    return {f"NASADEM_HGT_n{int(lat):02d}e{int(lon):03d}.zip"
            for lat, lon in zip(grade["latitude"], grade["longitude"])}


def coletar(etapa):
    if not GRADE.exists():
        criar_grade()
    if etapa == "mascaras":
        from modelo.coletar_mascaras import coletar as buscar
        return buscar("2025-01-01", "2025-12-31", PASTA, trabalhadores=3, area=AREA, tiles=TILES)
    if etapa == "firms":
        from modelo.coletar_firms import coletar as buscar
        return buscar("2024-12-24", "2025-12-31", PASTA, trabalhadores=3, area=AREA)
    if etapa == "clima":
        from modelo.coletar_clima import coletar as buscar
        return buscar("2024-12-22", "2025-12-31", PASTA, GRADE, trabalhadores=3)
    if etapa == "geografia":
        from modelo.coletar_geografia import coletar as buscar
        return buscar("2024-12-01", "2025-12-31", PASTA, trabalhadores=3, area=AREA, tiles=TILES, nomes_terreno=nomes_terreno())
    raise ValueError("Etapa de coleta desconhecida.")


def main():
    parser = argparse.ArgumentParser(description="Prepara um teste geografico independente na Suica.")
    parser.add_argument("etapa", choices=["grade", "mascaras", "firms", "clima", "geografia"])
    args = parser.parse_args()
    resultado = criar_grade() if args.etapa == "grade" else coletar(args.etapa)
    print(json.dumps({k: v for k, v in resultado.items() if k not in {"respostas", "lacunas"}}, ensure_ascii=False, indent=2))
    if resultado.get("falhas"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
