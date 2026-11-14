# Estado antes da evolução do modelo

Registro criado em 14/11/2026 antes de modificar o projeto. A cópia integral está em `../HACKATON-NASA-2_BACKUP_20261114_011539/`; os 151 arquivos foram comparados por SHA-256 e nenhuma diferença foi encontrada. A cópia inclui arquivos locais sensíveis e deve permanecer fora do Git.

- Branch e commit de referência: `main`, `79f508c`, `feat: add experimental VIIRS prediction model`.
- O diretório de trabalho já tinha alterações locais em `extratores/funcao_extrair.py`, `tratamento/config.py` e `tratamento/limpeza_tabelas.py`, além de dados, código e documentação ainda não versionados. Tudo foi preservado na cópia.
- O plano recebido em `Downloads/PLANO_EVOLUCAO_MODELO.md` era idêntico ao plano local.
- O artefato anterior permanece em `modelo/artefatos/viirs_regional/`. Seu teste histórico tinha 131 exemplos e 14 positivos: acurácia 82,44%, precisão 30,43%, revocação 50,00% e Brier 0,0858. A avaliação era exploratória e condicionada à máscara integralmente válida.
- No teste de sensibilidade sem históricos de satélite, foram identificados 2 dos 14 casos positivos; Brier 0,0938, contra 0,0921 da referência sazonal.
- A consulta atual anterior usava clima NASA POWER e oito recortes selecionados no Brasil; o histórico recente MODIS/VIIRS ainda estava ausente. O sistema não era validado para alertas ou previsão global.

Os valores acima são o ponto de comparação anterior. Novas métricas só serão diretamente comparáveis se usarem o mesmo alvo, os mesmos exemplos e a mesma regra de avaliação.
