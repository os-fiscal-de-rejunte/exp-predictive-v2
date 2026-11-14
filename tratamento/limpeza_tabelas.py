from pathlib import Path

try:
    from .validacoes_tabelas import argumentos, carregar_e_validar, exibir_relatorio
except ImportError:
    from validacoes_tabelas import argumentos, carregar_e_validar, exibir_relatorio


def preparar(eventos, celulas):
    eventos = eventos.drop_duplicates().sort_values("acq_timestamp_utc").reset_index(drop=True)
    celulas = celulas.sort_values(["grid_id", "acq_date"]).reset_index(drop=True)
    celulas["ano"] = celulas["acq_date"].dt.year
    celulas["mes"] = celulas["acq_date"].dt.month
    celulas["dia_do_ano"] = celulas["acq_date"].dt.dayofyear
    return eventos, celulas


def main():
    parser = argumentos()
    parser.add_argument("--saida", type=Path, default=Path(__file__).resolve().parent.parent / "dados")
    args = parser.parse_args()
    eventos, celulas, rel_eventos, rel_celulas = carregar_e_validar(args.eventos, args.celulas)
    exibir_relatorio("eventos", eventos, rel_eventos)
    exibir_relatorio("celulas", celulas, rel_celulas)
    if any(not r.empty and r["bloqueante"].any() for r in (rel_eventos, rel_celulas)):
        raise SystemExit("Preparação interrompida: corrija as inconsistências bloqueantes.")
    eventos, celulas = preparar(eventos, celulas)
    args.saida.mkdir(parents=True, exist_ok=True)
    eventos.to_csv(args.saida / "eventos_preparados.csv", index=False, encoding="utf-8")
    celulas.to_csv(args.saida / "celulas_preparadas.csv", index=False, encoding="utf-8")
    print(f"\nEventos preparados: {len(eventos):,}")
    print(f"Células preparadas: {len(celulas):,}")
    print("Os arquivos preservam valores ausentes e ainda não definem alvo ou entradas para treinamento.")


if __name__ == "__main__":
    main()
