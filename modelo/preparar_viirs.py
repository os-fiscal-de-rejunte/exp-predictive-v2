import argparse
import json

from modelo.viirs_dados import BASE, METADADOS, ORIGEM, preparar


def main():
    parser = argparse.ArgumentParser(description="Prepara uma copia auditada da amostra diaria VIIRS.")
    parser.add_argument("--origem", default=str(ORIGEM))
    parser.add_argument("--saida", default=str(BASE))
    parser.add_argument("--metadados", default=str(METADADOS))
    args = parser.parse_args()
    print(json.dumps(preparar(args.origem, args.saida, args.metadados), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
