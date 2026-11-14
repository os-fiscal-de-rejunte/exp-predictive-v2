import argparse
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path

from modelo.coletar_clima import coletar as coletar_clima
from modelo.coletar_firms import coletar as coletar_firms
from modelo.coletar_geografia import coletar as coletar_geografia
from modelo.coletar_mascaras import PASTA, coletar as coletar_mascaras, sha256
from modelo.geografia_grade import processar as processar_geografia
from modelo.grade import SAIDA as GRADE
from modelo.mascaras import processar as processar_mascaras
from modelo.preparar_grade_atual import PASTA as PASTA_ATUAL, preparar
from modelo.prever_grade import SAIDA, geojson, prever
from modelo.registro_prospectivo import registrar
from modelo.treinar_grade import SAIDA as ARTEFATOS


def pasta_geografia(historica=PASTA, atual=PASTA_ATUAL):
    historica, atual = Path(historica), Path(atual)
    recente = atual / "geografia"
    if recente.exists():
        return recente
    manifesto = historica / "manifesto_geografia.json"
    if manifesto.exists():
        dados = json.loads(manifesto.read_text(encoding="utf-8"))
        arquivos = dados.get("arquivos", {})
        if arquivos and all((historica / "brutos" / "geografia" / nome).is_file() for nome in arquivos):
            return historica
    return recente


def geografia_atualizada(agora, pasta=PASTA):
    pasta = Path(pasta)
    resumo = pasta / "resumo_geografia_atual.json"
    vegetacao = pasta / "vegetacao_grade_atual.parquet"
    relevo = pasta / "relevo_grade_atual.parquet"
    if not all(p.exists() for p in [resumo, vegetacao, relevo]):
        return False
    criado = datetime.fromtimestamp(resumo.stat().st_mtime, timezone.utc)
    if agora - criado > timedelta(days=7) or criado > agora:
        return False
    dados = json.loads(resumo.read_text(encoding="utf-8"))
    return dados.get("sha256_vegetacao") == sha256(vegetacao) and dados.get("sha256_relevo") == sha256(relevo)


def atualizar(saida=SAIDA):
    agora = datetime.now(timezone.utc)
    alvo = agora.date() + timedelta(days=1)
    inicio = alvo - timedelta(days=17)
    fim = alvo - timedelta(days=2)
    inicio_clima = alvo - timedelta(days=23)
    fim_clima = alvo - timedelta(days=3)
    print("Atualizando mascaras VIIRS disponiveis...", flush=True)
    mascaras = coletar_mascaras(inicio.isoformat(), fim.isoformat(), PASTA_ATUAL)
    if mascaras["falhas"]:
        raise RuntimeError("Falha na coleta de mascaras; nenhuma previsao foi emitida.")
    print("Atualizando historico FIRMS NRT...", flush=True)
    coletar_firms(inicio.isoformat(), fim.isoformat(), PASTA_ATUAL, fontes=["VIIRS_NOAA20_NRT", "MODIS_NRT"])
    print("Atualizando clima NASA POWER...", flush=True)
    coletar_clima(inicio_clima.isoformat(), fim_clima.isoformat(), PASTA_ATUAL)
    pasta_geo = pasta_geografia()
    if not geografia_atualizada(datetime.now(timezone.utc), pasta_geo):
        inicio_geo = (alvo - timedelta(days=100)).replace(day=1)
        print("Atualizando vegetacao e relevo NASA...", flush=True)
        resultado_geo = coletar_geografia(inicio_geo.isoformat(), fim.isoformat(), pasta=pasta_geo)
        if resultado_geo["falhas"]:
            raise RuntimeError("Falha na coleta geografica; nenhuma previsao foi emitida.")
        processar_geografia(pasta=pasta_geo, sufixo="_atual")
    if datetime.now(timezone.utc).date() != agora.date():
        raise RuntimeError("A data UTC mudou durante a coleta; reinicie a consulta.")
    print("Preparando entradas e previsoes...", flush=True)
    processar_mascaras(inicio.isoformat(), alvo.isoformat(), PASTA_ATUAL)
    arquivo_mascaras = PASTA_ATUAL / f"mascaras_{inicio:%Y%m%d}_{alvo:%Y%m%d}.parquet"
    arquivo_firms = PASTA_ATUAL / f"firms_{inicio:%Y%m%d}_{fim:%Y%m%d}.parquet"
    arquivo_clima = PASTA_ATUAL / f"clima_{inicio_clima:%Y%m%d}_{fim_clima:%Y%m%d}.parquet"
    arquivo_veg = pasta_geo / "vegetacao_grade_atual.parquet"
    arquivo_relevo = pasta_geo / "relevo_grade_atual.parquet"
    preparado = preparar(arquivo_mascaras, arquivo_clima, arquivo_firms, arquivo_veg, arquivo_relevo)
    arquivo_entrada = PASTA_ATUAL / f"caracteristicas_atuais_{alvo:%Y%m%d}.parquet"
    resultado = prever(arquivo_entrada, ARTEFATOS / "modelo_grade.joblib", ARTEFATOS / "relatorio_grade.json", GRADE)
    registro = registrar(resultado)
    destino = Path(saida)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    destino.with_suffix(".geojson").write_text(json.dumps(geojson(resultado), ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return {"alvo_utc": alvo.isoformat(), "celulas": len(resultado["celulas"]),
            "status": resultado["status"], "saida": str(destino), "geojson": str(destino.with_suffix(".geojson")),
            "registro_prospectivo": str(registro),
            "mascaras_ausentes_no_catalogo": len(mascaras["lacunas"]), "sha256_entradas": preparado["sha256_saida"]}


def main():
    parser = argparse.ArgumentParser(description="Atualiza dados e gera a previsao experimental da grade regional.")
    parser.add_argument("--saida", type=Path, default=SAIDA)
    args = parser.parse_args()
    print(json.dumps(atualizar(args.saida), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
