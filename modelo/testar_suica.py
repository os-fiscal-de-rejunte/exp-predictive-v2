import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from zipfile import ZipFile

import joblib
from netCDF4 import Dataset
import numpy as np
import pandas as pd

from modelo.caracteristicas_grade import CLIMA, adicionar_vegetacao, clima_janelas
from modelo.coletar_clima import celulas_meteorologicas
from modelo.coletar_mascaras import sha256
from modelo.mascaras import RAIO, VALIDAS, FOGO
from modelo.suica import GRADE, PASTA
from modelo.treinar_grade import SAIDA, probabilidade


def mapa_pixels(x, y, grade):
    latitudes = np.rad2deg(y / RAIO)
    linhas_uteis = np.flatnonzero((latitudes >= 45.75) & (latitudes < 47.75))
    lat = latitudes[linhas_uteis, None]
    lon = np.rad2deg(x[None, :] / (RAIO * np.cos(np.deg2rad(lat))))
    dentro = (lon >= 5.75) & (lon < 10.75)
    r, c = np.nonzero(dentro)
    linha = np.floor((latitudes[linhas_uteis[r]] - 45.75) / 0.25).astype(int)
    coluna = np.floor((lon[r, c] - 5.75) / 0.25).astype(int)
    indices = np.full((8, 20), -1, dtype=int)
    indices[grade["linha"].to_numpy(), grade["coluna"].to_numpy()] = np.arange(len(grade))
    ids = indices[linha, coluna]
    aceitos = ids >= 0
    return linhas_uteis[r[aceitos]], c[aceitos], ids[aceitos]


def abrir_viirs(caminho, grade, produto):
    arquivo = Dataset(os.path.relpath(caminho, Path.cwd()))
    if arquivo.ShortName != produto or str(arquivo.VersionID) != "002" or arquivo.PlatformShortName != "JPSS-1":
        arquivo.close()
        raise ValueError("Produto VIIRS ou plataforma inesperados.")
    nome = "VIIRS_Grid_Daily_Fire" if produto == "VJ114A1" else "VIIRS_Grid_16Day_VI_1km"
    grupo = arquivo.groups["HDFEOS"].groups["GRIDS"].groups[nome]
    campos = grupo.groups["Data Fields"]
    projecao = campos.variables["Projection"]
    if projecao.grid_mapping_name != "sinusoidal" or abs(float(projecao.earth_radius) - RAIO) > 0.01:
        arquivo.close()
        raise ValueError("Projecao VIIRS inesperada.")
    x = np.asarray(grupo.variables["XDim"][:], dtype=float)
    y = np.asarray(grupo.variables["YDim"][:], dtype=float)
    if len(x) != 1200 or len(y) != 1200 or not np.allclose(np.diff(x), np.diff(x)[0], atol=0.02) or not np.allclose(np.diff(y), np.diff(y)[0], atol=0.02):
        arquivo.close()
        raise ValueError("Grade VIIRS inesperada.")
    x = x + np.diff(x)[0] / 2
    y = y + np.diff(y)[0] / 2
    return arquivo, campos, mapa_pixels(x, y, grade), (x[[0, -1]], y[[0, -1]])


def processar_mascaras(grade):
    manifesto = json.loads((PASTA / "manifesto_viirs.json").read_text(encoding="utf-8"))
    consultas = [x for x in manifesto["consultas"] if x["inicio"] == "2025-01-01" and x["fim"] == "2025-12-31"]
    if not consultas or consultas[-1]["falhas"]:
        raise ValueError("A coleta completa das mascaras suicas ainda nao foi concluida.")
    itens = {v["data"]: v for v in manifesto["arquivos"].values() if "2025-01-01" <= v["data"] <= "2025-12-31"}
    if not itens or len(itens) != sum("2025-01-01" <= v["data"] <= "2025-12-31" for v in manifesto["arquivos"].values()):
        raise ValueError("Granulos diarios ausentes ou duplicados.")
    origem = PASTA / "brutos" / "viirs"
    primeiro = itens[min(itens)]
    arquivo, _, mapa, coordenadas = abrir_viirs(origem / primeiro["nome"], grade, "VJ114A1")
    arquivo.close()
    esperados = np.bincount(mapa[2], minlength=len(grade))
    if (esperados == 0).any():
        raise ValueError("A grade suica nao esta inteiramente coberta pelo tile VIIRS.")
    linhas = []
    for numero, data in enumerate(pd.date_range("2025-01-01", "2025-12-31", freq="D"), 1):
        dia = data.date().isoformat()
        contagens = np.zeros((len(grade), 10), dtype=np.int64)
        producao = None
        if dia in itens:
            item = itens[dia]
            caminho = origem / item["nome"]
            if sha256(caminho) != item["sha256"]:
                raise ValueError("Granulo VIIRS da Suica alterado.")
            arquivo, campos, mapa_atual, pontos = abrir_viirs(caminho, grade, "VJ114A1")
            try:
                if arquivo.RangeBeginningDate != dia or not all(np.allclose(a, b, atol=0.02) for a, b in zip(coordenadas, pontos)):
                    raise ValueError("Data ou coordenadas do granulo VIIRS divergentes.")
                if not all(np.array_equal(a, b) for a, b in zip(mapa, mapa_atual)):
                    raise ValueError("Mapeamento espacial VIIRS divergente.")
                valores = np.ma.filled(np.ma.asarray(campos.variables["FireMask"][:], dtype=np.int16), -1)
                amostra = valores[mapa[0], mapa[1]]
                for classe in range(10):
                    contagens[:, classe] = np.bincount(mapa[2][amostra == classe], minlength=len(grade))
                producao = str(arquivo.ProductionTime)
            finally:
                arquivo.close()
        validos = contagens[:, VALIDAS].sum(axis=1)
        fogo = contagens[:, FOGO].sum(axis=1)
        completa = validos == esperados
        quadro = grade[["grid_id", "latitude", "longitude", "linha", "coluna"]].copy()
        quadro["data_utc"] = dia
        quadro["pixels_esperados"] = esperados
        quadro["pixels_validos"] = validos
        quadro["pixels_fogo"] = fogo
        quadro["pixels_nuvem"] = contagens[:, 4]
        quadro["mascara_completa"] = completa
        quadro["evidencia_foco"] = fogo > 0
        quadro["alvo_condicional"] = pd.array(np.where(completa, (fogo > 0).astype(float), np.nan), dtype="Int64")
        quadro["producao_utc"] = producao
        linhas.append(quadro)
        if numero % 30 == 0 or numero == 365:
            print(f"Dias de mascara processados: {numero}/365", flush=True)
    dados = pd.concat(linhas, ignore_index=True)
    caminho = PASTA / "mascaras_suica_2025.parquet"
    dados.to_parquet(caminho, index=False)
    completas = dados["mascara_completa"]
    resumo = {"linhas": len(dados), "completas": int(completas.sum()),
              "positivas_completas": int((completas & dados["evidencia_foco"]).sum()),
              "incompletas": int((~completas).sum()),
              "positivas_incompletas": int((~completas & dados["evidencia_foco"]).sum()),
              "dias_sem_granulo": 365 - len(itens), "sha256_arquivo": sha256(caminho),
              "sha256_manifesto": sha256(PASTA / "manifesto_viirs.json")}
    (PASTA / "resumo_mascaras_suica_2025.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    return resumo


def processar_vegetacao(grade, itens, origem):
    quadros = []
    for item in sorted(itens, key=lambda x: x["nome"]):
        caminho = origem / item["nome"]
        if sha256(caminho) != item["sha256"]:
            raise ValueError("Granulo de vegetacao alterado.")
        arquivo, campos, mapa, _ = abrir_viirs(caminho, grade, "VJ113A2")
        try:
            arrays = []
            for nome in ["1 km 16 days NDVI", "1 km 16 days EVI", "1 km 16 days VI Quality", "1 km 16 days pixel reliability"]:
                campo = campos.variables[nome]
                campo.set_auto_maskandscale(False)
                arrays.append(np.asarray(campo[:])[mapa[0], mapa[1]])
            ndvi, evi, qa, confiabilidade = arrays
            bom = ((qa & 3) == 0) & ((qa & (1 << 8)) == 0) & ((qa & (1 << 10)) == 0)
            bom &= ((qa & (3 << 14)) == 0) & (((qa >> 11) & 7) == 1)
            bom &= (confiabilidade >= 0) & (confiabilidade <= 2)
            bom &= (ndvi >= -10000) & (ndvi <= 10000) & (evi >= -10000) & (evi <= 10000)
            n = np.bincount(mapa[2][bom], minlength=len(grade))
            esperados = np.bincount(mapa[2], minlength=len(grade))
            if (esperados == 0).any():
                raise ValueError("Celula suica sem pixels de vegetacao esperados.")
            fracao = n / esperados
            valido = (n > 0) & (fracao >= 0.2)
            ndvi_soma = np.bincount(mapa[2][bom], weights=ndvi[bom].astype(float) * 0.0001, minlength=len(grade))
            evi_soma = np.bincount(mapa[2][bom], weights=evi[bom].astype(float) * 0.0001, minlength=len(grade))
            fim = pd.to_datetime(str(arquivo.RangeEndingDate), utc=True)
            disponivel = max(pd.to_datetime(str(arquivo.ProductionTime), utc=True), fim + pd.Timedelta(days=1))
            quadros.append(pd.DataFrame({"grid_id": grade["grid_id"], "composto_fim": fim,
                                        "disponivel_utc": disponivel,
                                        "ndvi": np.divide(ndvi_soma, n, out=np.full(len(grade), np.nan), where=valido),
                                        "evi": np.divide(evi_soma, n, out=np.full(len(grade), np.nan), where=valido),
                                        "frac_qualidade": fracao}))
        finally:
            arquivo.close()
    if not quadros:
        raise ValueError("Nenhum composto de vegetacao para a Suica.")
    return pd.concat(quadros, ignore_index=True)


def processar_relevo(grade, itens, origem):
    valores = []
    for item in itens:
        correspondencia = re.fullmatch(r"NASADEM_HGT_n(\d{2})e(\d{3})\.zip", item["nome"])
        if not correspondencia:
            raise ValueError("Tile de relevo suico inesperado.")
        sul, oeste = map(int, correspondencia.groups())
        caminho = origem / item["nome"]
        if sha256(caminho) != item["sha256"]:
            raise ValueError("Tile de relevo suico alterado.")
        with ZipFile(caminho) as arquivo:
            membros = [nome for nome in arquivo.namelist() if nome.lower().endswith(".hgt")]
            if len(membros) != 1:
                raise ValueError("Arquivo de relevo sem raster unico.")
            bruto = arquivo.read(membros[0])
        if len(bruto) != 3601 * 3601 * 2:
            raise ValueError("Raster NASADEM com dimensao inesperada.")
        altura = np.frombuffer(bruto, dtype=">i2").reshape(3601, 3601)
        latitudes = sul + 1 - np.arange(3601) / 3600
        longitudes = oeste + np.arange(3601) / 3600
        celulas = grade[(grade["latitude"].astype(int) == sul) & (grade["longitude"].astype(int) == oeste)]
        for celula in celulas.itertuples(index=False):
            linhas = np.flatnonzero((latitudes >= celula.sul) & (latitudes < celula.norte))
            colunas = np.flatnonzero((longitudes >= celula.oeste) & (longitudes < celula.leste))
            bloco = altura[np.ix_(linhas, colunas)]
            valido = bloco[(bloco >= -500) & (bloco <= 9000)].astype(float)
            valores.append({"grid_id": celula.grid_id, "altitude_media_m": float(valido.mean()) if len(valido) else np.nan,
                            "altitude_desvio_m": float(valido.std()) if len(valido) else np.nan,
                            "altitude_pixels_validos": len(valido)})
    dados = pd.DataFrame(valores)
    if len(dados) != len(grade) or dados["grid_id"].duplicated().any():
        raise ValueError("Relevo nao cobre exatamente a grade suica.")
    return dados


def processar_geografia(grade):
    manifesto = json.loads((PASTA / "manifesto_geografia.json").read_text(encoding="utf-8"))
    itens = list(manifesto["arquivos"].values())
    origem = PASTA / "brutos" / "geografia"
    vegetacao = processar_vegetacao(grade, [x for x in itens if x["produto"] == "VJ113A2"], origem)
    relevo = processar_relevo(grade, [x for x in itens if x["produto"] == "NASADEM_HGT"], origem)
    vegetacao_path = PASTA / "vegetacao_suica.parquet"
    relevo_path = PASTA / "relevo_suica.parquet"
    vegetacao.to_parquet(vegetacao_path, index=False)
    relevo.to_parquet(relevo_path, index=False)
    resumo = {"compostos": int(vegetacao["composto_fim"].nunique()), "linhas_vegetacao": len(vegetacao),
              "ndvi_valido": int(vegetacao["ndvi"].notna().sum()), "celulas_relevo": len(relevo),
              "sha256_vegetacao": sha256(vegetacao_path), "sha256_relevo": sha256(relevo_path),
              "sha256_manifesto": sha256(PASTA / "manifesto_geografia.json")}
    (PASTA / "resumo_geografia_suica.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    return resumo


def adicionar_firms(dados, detecoes, grade):
    firmas = detecoes.copy()
    firmas["instante_utc"] = pd.to_datetime(firmas["instante_utc"], utc=True, errors="raise")
    firmas["linha"] = np.floor((firmas["latitude"] - 45.75) / 0.25).astype(int)
    firmas["coluna"] = np.floor((firmas["longitude"] - 5.75) / 0.25).astype(int)
    firmas = firmas.merge(grade[["linha", "coluna", "grid_id"]], on=["linha", "coluna"], how="inner", validate="many_to_one")
    firmas["data_utc"] = firmas["instante_utc"].dt.normalize()
    firmas["tipo"] = np.where(firmas["fonte"].str.startswith("VIIRS_NOAA20"), "viirs", "modis")
    grupo = firmas.groupby(["grid_id", "data_utc", "tipo"], sort=False)
    totais = pd.concat([grupo.size().rename("registros"), grupo["frp"].sum(min_count=1).rename("frp")], axis=1).reset_index()
    saida = dados.sort_values(["grid_id", "data_utc"]).copy()
    for tipo in ["viirs", "modis"]:
        parte = totais[totais["tipo"].eq(tipo)].drop(columns="tipo").rename(columns={"registros": "_registros", "frp": "_frp"})
        saida = saida.merge(parte, on=["grid_id", "data_utc"], how="left", validate="one_to_one", sort=False)
        agrupado = saida.groupby("grid_id", sort=False)
        for nome in ["registros", "frp"]:
            janela = pd.concat([agrupado["_" + nome].shift(i) for i in range(2, 9)], axis=1)
            saida[f"{tipo}_{nome}_7d"] = janela.sum(axis=1, min_count=1)
        saida = saida.drop(columns=["_registros", "_frp"])
    return saida


def preparar(grade):
    mascaras_path = PASTA / "mascaras_suica_2025.parquet"
    clima_path = PASTA / "clima_20241222_20251231.parquet"
    firms_path = PASTA / "firms_20241224_20251231.parquet"
    veg_path, relevo_path = PASTA / "vegetacao_suica.parquet", PASTA / "relevo_suica.parquet"
    verificacoes = [("resumo_mascaras_suica_2025.json", "sha256_arquivo", mascaras_path),
                    ("manifesto_clima_20241222_20251231.json", "sha256_arquivo", clima_path),
                    ("manifesto_firms_20241224_20251231.json", "sha256_arquivo", firms_path),
                    ("resumo_geografia_suica.json", "sha256_vegetacao", veg_path),
                    ("resumo_geografia_suica.json", "sha256_relevo", relevo_path),
                    ("manifesto_grade.json", "sha256_grade", GRADE)]
    for manifesto, campo, arquivo in verificacoes:
        esperado = json.loads((PASTA / manifesto).read_text(encoding="utf-8"))[campo]
        if esperado != sha256(arquivo):
            raise ValueError(f"Arquivo divergente do manifesto: {arquivo.name}.")
    mascaras = pd.read_parquet(mascaras_path)
    mascaras["data_utc"] = pd.to_datetime(mascaras["data_utc"], utc=True, errors="raise")
    base = grade[["grid_id", "latitude", "longitude"]].merge(
        pd.DataFrame({"data_utc": pd.date_range("2024-12-24", "2025-12-31", freq="D", tz="UTC")}), how="cross")
    base = base.merge(mascaras[["grid_id", "data_utc", "mascara_completa", "alvo_condicional", "evidencia_foco"]],
                      on=["grid_id", "data_utc"], how="left", validate="one_to_one")
    base = adicionar_firms(base, pd.read_parquet(firms_path), grade)
    base = base.merge(celulas_meteorologicas(grade)[["grid_id", "lat_power", "lon_power"]], on="grid_id", validate="many_to_one")
    base = base.merge(clima_janelas(pd.read_parquet(clima_path)), on=["lat_power", "lon_power", "data_utc"], how="left", validate="many_to_one")
    base = adicionar_vegetacao(base, pd.read_parquet(veg_path))
    base = base.merge(pd.read_parquet(relevo_path), on="grid_id", how="left", validate="many_to_one")
    dia = base["data_utc"].dt.dayofyear
    base["dia_seno"] = np.sin(2 * np.pi * (dia - 1) / 365.0)
    base["dia_cosseno"] = np.cos(2 * np.pi * (dia - 1) / 365.0)
    caminho = PASTA / "caracteristicas_suica_2025.parquet"
    base.to_parquet(caminho, index=False)
    return base, {"mascaras": sha256(mascaras_path), "clima": sha256(clima_path), "firms": sha256(firms_path),
                  "vegetacao": sha256(veg_path), "relevo": sha256(relevo_path), "grade": sha256(GRADE), "caracteristicas": sha256(caminho)}


def avaliar_medidas(y, p, classes):
    from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, precision_score, recall_score
    y, p, classes = np.asarray(y, dtype=int), np.asarray(p, dtype=float), np.asarray(classes, dtype=int)
    positivos, previstos = int(y.sum()), int(classes.sum())
    return {"linhas": len(y), "positivos": positivos, "positivos_previstos": previstos,
            "acuracia": float(accuracy_score(y, classes)),
            "precisao": float(precision_score(y, classes, zero_division=0)) if previstos else None,
            "revocacao": float(recall_score(y, classes, zero_division=0)) if positivos else None,
            "average_precision": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None,
            "brier": float(brier_score_loss(y, p)),
            "matriz_confusao": confusion_matrix(y, classes, labels=[0, 1]).tolist()}


def executar(grade):
    dados, fontes = preparar(grade)
    modelo_path, relatorio_path = SAIDA / "modelo_grade.joblib", SAIDA / "relatorio_grade.json"
    artefato = joblib.load(modelo_path)
    relatorio_modelo = json.loads(relatorio_path.read_text(encoding="utf-8"))
    if artefato["sha256_dados"] != relatorio_modelo["sha256_dados"]:
        raise ValueError("Modelo e relatorio brasileiros divergentes.")
    dados = dados[dados["data_utc"].between(pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2025-12-31", tz="UTC"))].copy()
    completo = dados["mascara_completa"].astype("boolean").fillna(False).astype(bool)
    clima = dados[["clima_" + c for c in CLIMA]].notna().all(axis=1)
    geografia = dados[["vegetacao_ndvi", "vegetacao_evi", "altitude_media_m", "altitude_desvio_m"]].notna().all(axis=1)
    testavel = dados[completo & clima].copy()
    if testavel.empty:
        raise ValueError("Nenhuma celula/dia com mascara completa e clima valido.")
    testavel["modo_modelo"] = np.where(geografia.loc[testavel.index], "firms_sem_mascara", "somente_clima")
    testavel["probabilidade"] = np.nan
    testavel["classe"] = 0
    testavel["limiar"] = np.nan
    for modo in ["firms_sem_mascara", "somente_clima"]:
        filtro = testavel["modo_modelo"].eq(modo)
        if not filtro.any():
            continue
        pacote = artefato["modos"][modo]
        p = probabilidade(pacote["modelo"], pacote["calibrador"], testavel.loc[filtro], pacote["colunas"])
        testavel.loc[filtro, "probabilidade"] = p
        testavel.loc[filtro, "classe"] = (p >= pacote["limiar"]).astype(int)
        testavel.loc[filtro, "limiar"] = float(pacote["limiar"])
    y = testavel["alvo_condicional"].astype(int).to_numpy()
    p = testavel["probabilidade"].to_numpy(dtype=float)
    classes = testavel["classe"].to_numpy(dtype=int)
    medidas = avaliar_medidas(y, p, classes)
    por_modo = {}
    for modo, grupo in testavel.groupby("modo_modelo"):
        por_modo[modo] = avaliar_medidas(grupo["alvo_condicional"], grupo["probabilidade"], grupo["classe"])
        por_modo[modo]["limiar"] = float(artefato["modos"][modo]["limiar"])
        por_modo[modo]["maior_probabilidade"] = float(grupo["probabilidade"].max())
    por_mes = []
    for mes, grupo in testavel.groupby(testavel["data_utc"].dt.strftime("%Y-%m")):
        por_mes.append({"mes": mes, "linhas": len(grupo), "positivos": int(grupo["alvo_condicional"].sum()),
                        "acertos_positivos": int(((grupo["alvo_condicional"] == 1) & (grupo["classe"] == 1)).sum())})
    treino_path = PASTA.parent / "evolucao" / "caracteristicas_grade.parquet"
    if sha256(treino_path) != artefato["sha256_dados"]:
        raise ValueError("Dados de referencia nao correspondem ao treinamento brasileiro.")
    referencia = pd.read_parquet(treino_path)
    referencia["data_utc"] = pd.to_datetime(referencia["data_utc"], utc=True)
    referencia["producao_utc"] = pd.to_datetime(referencia["producao_utc"], utc=True)
    referencia = referencia[(referencia["coluna"] <= 9) & referencia["mascara_completa"] &
                            (referencia["data_utc"] <= pd.Timestamp("2025-04-30", tz="UTC")) &
                            (referencia["producao_utc"] <= pd.Timestamp("2025-04-30T23:00:00Z"))]
    extrapolacao = {}
    for coluna in ["latitude", "longitude", "clima_T2M", "clima_RH2M", "clima_PRECTOTCORR",
                   "vegetacao_ndvi", "altitude_media_m"]:
        anterior = pd.to_numeric(referencia[coluna], errors="coerce").dropna()
        atual = pd.to_numeric(testavel[coluna], errors="coerce").dropna()
        if not anterior.empty and not atual.empty:
            inferior, superior = float(anterior.min()), float(anterior.max())
            extrapolacao[coluna] = {"min_treino": inferior, "max_treino": superior,
                                    "min_suica": float(atual.min()), "max_suica": float(atual.max()),
                                    "fracao_fora_intervalo_treino": float(((atual < inferior) | (atual > superior)).mean())}
    saida = PASTA / "previsoes_teste_suica_2025.csv"
    testavel[["grid_id", "latitude", "longitude", "data_utc", "alvo_condicional", "modo_modelo", "probabilidade", "limiar", "classe"]].to_csv(saida, index=False, lineterminator="\r\n")
    relatorio = {"status": "exploratorio_inconclusivo" if medidas["positivos"] < 50 else "exploratorio_nao_operacional",
                 "criado_em_utc": datetime.now(timezone.utc).isoformat(), "pais": "Suica", "ano": 2025,
                 "populacao": {"celulas_internas": len(grade), "celulas_dia": len(dados),
                               "mascaras_completas": int(completo.sum()), "mascaras_incompletas": int((~completo).sum()),
                               "testaveis": len(testavel), "sem_clima": int((completo & ~clima).sum()),
                               "modo_firms": int(testavel["modo_modelo"].eq("firms_sem_mascara").sum()),
                               "modo_clima": int(testavel["modo_modelo"].eq("somente_clima").sum())},
                 "metricas": medidas, "por_modo": por_modo, "por_mes": por_mes,
                 "comparacao_intervalos_entrada": extrapolacao,
                 "referencia_sempre_sem_deteccao": {"acuracia": float(1 - y.mean()), "revocacao": 0.0, "brier": float(y.mean())},
                 "modelo_sha256": sha256(modelo_path), "fontes_sha256": fontes, "previsoes_sha256": sha256(saida),
                 "limites": ["Celulas inteiras internas da Suica, nao o pais todo.",
                             "Teste retrospectivo; POWER pode ter sido revisado e FIRMS SP pode diferir do NRT.",
                             "Modelo treinado no MATOPIBA; entradas podem estar fora do dominio de treinamento.",
                             "Rotulo condicionado a mascara VIIRS integralmente classificada; nao mede incendios em geral.",
                             "Menos de 50 positivos torna a avaliacao inconclusiva; celulas e dias sao dependentes."]}
    (PASTA / "relatorio_suica_2025.json").write_text(json.dumps(relatorio, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return relatorio


def main():
    parser = argparse.ArgumentParser(description="Processa e avalia o modelo regional brasileiro na Suica.")
    parser.add_argument("etapa", choices=["mascaras", "geografia", "avaliar"])
    args = parser.parse_args()
    grade = pd.read_csv(GRADE).sort_values(["linha", "coluna"]).reset_index(drop=True)
    funcoes = {"mascaras": processar_mascaras, "geografia": processar_geografia, "avaliar": executar}
    resultado = funcoes[args.etapa](grade)
    print(json.dumps(resultado, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
