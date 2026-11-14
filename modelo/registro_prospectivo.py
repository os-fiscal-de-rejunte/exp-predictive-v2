from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

from modelo.config import RAIZ

PASTA = RAIZ / "modelo" / "artefatos" / "evolucao_grade" / "previsoes_prospectivas"


def validar(resultado):
    emitido = datetime.fromisoformat(resultado["emitido_em_utc"].replace("Z", "+00:00"))
    inicio = datetime.fromisoformat(resultado["inicio_previsto_utc"].replace("Z", "+00:00"))
    fim = datetime.fromisoformat(resultado["fim_previsto_utc"].replace("Z", "+00:00"))
    if any(x.utcoffset() != timedelta(0) for x in [emitido, inicio, fim]):
        raise ValueError("As datas do registro devem estar em UTC.")
    if emitido >= inicio or inicio.time().isoformat() != "00:00:00" or fim - inicio != timedelta(days=1):
        raise ValueError("O registro deve preceder o inicio do dia previsto.")
    celulas = resultado["celulas"]
    if not celulas or len({c["celula_id"] for c in celulas}) != len(celulas):
        raise ValueError("Celulas ausentes ou duplicadas no registro.")
    for celula in celulas:
        p, limiar, classe = (celula.get(k) for k in ["probabilidade_condicional", "limiar_classificacao", "classe_detectada"])
        if celula["status"] == "dados_insuficientes":
            if any(x is not None for x in [p, limiar, classe]):
                raise ValueError("Celula sem dados possui classificacao.")
        elif p is None or limiar is None or not 0 <= p <= 1 or not 0 <= limiar <= 1 or classe != int(p >= limiar):
            raise ValueError("Previsao, limiar ou classe invalidos.")
    return emitido, inicio


def registrar(resultado, pasta=PASTA):
    emitido, inicio = validar(resultado)
    agora = datetime.now(timezone.utc)
    if emitido > agora or agora >= inicio or agora - emitido > timedelta(hours=12):
        raise ValueError("A previsao nao pode ser registrada fora da janela anterior ao alvo.")
    conteudo = json.dumps(resultado, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    digest = hashlib.sha256(conteudo).hexdigest()
    pasta = Path(pasta) / inicio.date().isoformat()
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / (emitido.strftime("%Y%m%dT%H%M%S%fZ_") + digest + ".json")
    try:
        with caminho.open("xb") as arquivo:
            arquivo.write(conteudo)
    except FileExistsError:
        if caminho.read_bytes() != conteudo:
            raise ValueError("Registro existente diverge do conteudo.")
    return caminho


def carregar(caminho):
    caminho = Path(caminho)
    conteudo = caminho.read_bytes()
    if caminho.stem.rsplit("_", 1)[-1] != hashlib.sha256(conteudo).hexdigest():
        raise ValueError("O conteudo do registro prospectivo foi alterado.")
    resultado = json.loads(conteudo)
    emitido, inicio = validar(resultado)
    if caminho.parent.name != inicio.date().isoformat() or not caminho.name.startswith(emitido.strftime("%Y%m%dT%H%M%S%fZ_")):
        raise ValueError("O nome do registro nao corresponde a emissao e ao alvo.")
    return resultado
