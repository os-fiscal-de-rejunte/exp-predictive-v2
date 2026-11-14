# Modelo experimental de detecção de focos

Este projeto contém modelos experimentais treinados com dados reais para estimar a detecção de pelo menos um pixel de foco em uma área, no próximo dia UTC. A estimativa é **condicionada a todos os pixels da futura máscara VIIRS estarem classificados**. Não é a probabilidade de incêndio, de ignição ou de haver algum foco no país inteiro.

## Evolução: grade regional do MATOPIBA

A versão mais recente cobre 256 células de 0,25° × 0,25° em um recorte do MATOPIBA, com dados históricos de 2024–2025. Ela combina detecções MODIS e VIIRS/FIRMS, clima histórico, vegetação e terreno quando disponíveis, e grava resultados em JSON e GeoJSON. A cobertura do solo aparece como contexto, com o ano de referência, mas não entra no modelo principal. Os arquivos descrevem datas, fontes, modo do modelo e condição da previsão. A área é um piloto limitado e não representa o Brasil inteiro.

No teste retrospectivo, o modo usado na consulta atual teve acurácia de **77,99%**, precisão de **36,68%** e revocação de **42,62%**. A análise é exploratória, os dados já foram examinados e a revocação ficou abaixo da meta. Os resultados e as limitações estão no [relatório de evolução](RELATORIO_EVOLUCAO.md). Não use o modelo para alertas ou decisões operacionais.

Com Python 3.11, no PowerShell, dentro da pasta do projeto, instale as dependências e atualize a previsão:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r modelo/requirements.txt
.\.venv\Scripts\python.exe -B -m modelo.atualizar_grade
```

A atualização precisa de `EARTHDATA_TOKEN` e `FIRMS_MAP_KEY` no `.env` local. Os resultados ficam em `modelo/artefatos/evolucao_grade/previsao_atual.json` e `previsao_atual.geojson`. Cada emissão também é guardada em `previsoes_prospectivas/`, com a data, o modelo e o conteúdo verificado por SHA-256. A execução é sob demanda; não há agendamento automático.

Para verificar previsões já encerradas, consultar as máscaras disponíveis e gerar uma avaliação prospectiva:

```powershell
.\.venv\Scripts\python.exe -B -m modelo.avaliar_prospectivo --atualizar-observacoes
```

O resultado inicial é `aguardando_fim_do_dia` ou `aguardando_mascaras`. A avaliação depende da publicação posterior das máscaras; um arquivo local e seu hash não provam o horário de emissão por um terceiro independente.

O modelo treinado, os relatórios e as bases históricas processadas necessárias para reproduzir as avaliações estão no repositório. Dados brutos, cache, consultas atuais e registros prospectivos permanecem locais e são ignorados pelo Git. Em outro computador, a atualização atual baixa as fontes necessárias novamente. Carregue artefatos Joblib somente de origem confiável e mantenha as credenciais no `.env` local.

## Teste na Suíça

O modelo brasileiro foi avaliado em 52 células internas da Suíça durante 2025. Com o limiar original, encontrou **0 dos 23 casos positivos** com máscara completa. A amostra é pequena, mas não sustenta o uso do modelo para a Suíça. O [relatório do teste](RELATORIO_TESTE_SUICA.md) apresenta a cobertura, as métricas e os comandos de reprodução.

## Versão anterior: oito recortes

O piloto cobre oito recortes selecionados no Brasil, cada um com 41 × 41 pixels de aproximadamente 927 m em projeção sinusoidal. As áreas não foram sorteadas para representar o Brasil e algumas podem abranger regiões próximas a fronteiras. A consulta por país devolve somente esses recortes, não um mapa nacional ou mundial. O projeto ainda não é apropriado para alertas ou decisões operacionais.

### Dados e rótulo

O produto-alvo é o [VJ114A1.002](https://doi.org/10.5067/VIIRS/VJ114A1.002), máscara diária VIIRS/NOAA-20 de resolução nominal de 1 km. Ele é diferente das detecções FIRMS VIIRS de 375 m. Na leitura usada aqui, as classes 3, 5, 7, 8 e 9 são classificadas; 7 a 9 indicam foco. Nuvens e pixels desconhecidos não são negativos.

A base reaproveita uma cópia processada em `HACKATON-NASA/runs/viirs_daily_v4b_20261111/processed/`, sem alterar os arquivos originais nem o banco. A preparação confere o SHA-256 da origem contra o manifesto anterior, verifica as classes e registra os nomes e hashes dos grânulos no CSV. O processamento anterior também verificou os grânulos e as bordas dos recortes; veja [o relatório de origem](../HACKATON-NASA/REPORT_VIIRS_V4.md) e [a metodologia](../HACKATON-NASA/docs/metodologia_viirs_v4.md).

| Situação da máscara | Registros | Uso no piloto |
| --- | ---: | --- |
| Completa, com foco | 57 | Rótulo positivo |
| Completa, sem foco | 517 | Rótulo de não detecção observada |
| Incompleta | 570 | Não entra na avaliação condicional |
| Incompleta, mas com foco detectado | 21, incluídos nas 570 | Evidência positiva preservada; fora deste alvo condicionado à máscara completa |

O conjunto reúne datas amostradas entre 2018 e 2025, principalmente dias 1, 8, 15 e 22 de janeiro, abril, julho e outubro, com datas adicionais em 2023. Não é uma série contínua. O condicionamento é necessário para interpretar o piloto de forma uniforme; uma detecção em máscara incompleta continua sendo evidência, e não é convertida em negativo.

O arquivo anterior de células FIRMS de 0,01° permanece separado. Seu código genérico em `modelo/dados.py` agora aceita positivos mesmo sem máscara e só usa zero quando há observação válida. Essa mistura de positivos e negativos pode produzir seleção enviesada: a saída desse caminho não é uma probabilidade incondicional para todas as células. A execução genérica exige que o operador confirme a disponibilidade temporal das entradas.

### Como executar

No PowerShell, entre nesta pasta `HACKATON-NASA-2` e instale as dependências:

```powershell
python -m pip install -r modelo/requirements.txt
```

A base preparada já está em `dados/base_viirs_regional.csv`. Para recriá-la a partir da origem local e conferir o manifesto:

```powershell
python -B -m modelo.preparar_viirs
```

Para treinar novamente, comparar modelos e salvar artefatos:

```powershell
python -B -m modelo.treinar_viirs
```

Os resultados são gravados em `modelo/artefatos/viirs_regional/`: o modelo `modelo_viirs.joblib`, o relatório `relatorio_viirs.json` e as previsões individuais do teste. Esses três arquivos estão versionados e podem ser recriados; consultas atuais são geradas localmente. Carregue arquivos Joblib somente de fonte confiável e, após trocar versões de bibliotecas, treine novamente.

Para reproduzir uma consulta histórica, informe a data de referência, um dia antes da data prevista:

```powershell
python -B -m modelo.prever_viirs --pais Brasil --data 2025-10-21 --regiao centro_norte
```

Essa consulta aceita somente a base auditada associada ao modelo. Outras datas não são inventadas. Omitir `--regiao` retorna os recortes que existem no dia selecionado.

Para consultar o próximo dia UTC usando clima recente NASA POWER:

```powershell
python -B -m modelo.prever_atual --pais Brasil --regiao centro_norte
```

Omitir `--regiao` consulta os oito recortes. Para guardar o JSON, acrescente `--saida caminho/resultado.json`. A consulta registra o horário, as entradas meteorológicas e os hashes do modelo, da base e da resposta POWER. Se faltar algum dia ou parâmetro climático, aquele recorte retorna `dados_indisponiveis`, sem probabilidade.

**Evolução da consulta dos recortes:** a rotina busca o histórico recente FIRMS e usa modos treinados separadamente conforme a disponibilidade dos dados. MODIS é apresentado como contexto quando não corresponde a uma entrada histórica equivalente. A saída continua experimental e não libera uma probabilidade incondicional. A avaliação original e o teste de sensibilidade estão descritos abaixo.

### Como o modelo anterior foi avaliado

A emissão retrospectiva é considerada no fim de D−1 UTC, para o alvo em D. As entradas são latitude, longitude, sazonalidade, registros anteriores VIIRS/FIRMS, máscara MODIS anterior, máscara VIIRS anterior quando disponível e clima histórico POWER. O clima agrega oito dias de D−10 a D−3: soma da precipitação e média das demais variáveis. São verificadas as datas de produção das máscaras anteriores e as datas finais das janelas. Os arquivos retrospectivos podem ter sido revisados, portanto isso não comprova que todas as entradas estavam publicadas no momento histórico da emissão. A [NASA informa latência variável no POWER](https://power.larc.nasa.gov/docs/faqs/data/); o produto é histórico, não previsão meteorológica futura.

O ajuste usou seis áreas e exemplos de 2018 a 2021. Regressão logística, floresta e frequência constante foram comparadas por Brier em dobras cronológicas; a logística foi escolhida. Os dados de 2022 foram usados para calibrar e os de 2023 para decidir se manter a calibração e escolher o limiar por F1. Essas etapas têm apenas quatro e cinco positivos, respectivamente, e são instáveis. Os testes de 2024–2025 incluem as seis áreas de ajuste e outras duas áreas reservadas espacialmente. Os mesmos dados já haviam sido explorados no outro repositório, portanto não constituem teste cego prospectivo.

| Medida no teste histórico | Resultado |
| --- | ---: |
| Exemplos / positivos | 131 / 14 |
| Acurácia | 82,44% |
| Precisão | 30,43% |
| Revocação | 50,00% |
| Average precision | 0,3641 |
| Brier, menor é melhor | 0,0858 |
| Matriz `[[verdadeiros negativos, falsos positivos], [falsos negativos, verdadeiros positivos]]` | `[[101, 16], [7, 7]]` |

Uma regra que classificasse todos como negativos teria acurácia de 89,31%, mas revocação de 0%. Na comparação das probabilidades, a frequência global obteve Brier 0,0955 e a referência sazonal 0,0921. Os classificadores de referência são mostrados no relatório com limiar 0,5; o limiar do modelo foi escolhido na etapa de decisão. A análise espacial reservada tem apenas 35 exemplos e cinco positivos, insuficientes para generalizar a novas regiões.

Uma análise posterior zerou **a disponibilidade dos três históricos de satélite**, mantendo-os ausentes em vez de inventar registros. Nos mesmos 131 exemplos, o modelo teve Brier 0,0938 e encontrou 2 de 14 positivos no limiar já escolhido; a referência sazonal teve Brier 0,0921. Essa análise motivou as restrições da consulta atual e também não é uma avaliação prospectiva.

### Organização anterior

- `modelo/viirs_dados.py`: valida o contrato, os rótulos, datas, geometrias e proveniência da base.
- `modelo/preparar_viirs.py`: cria a cópia auditada e seu arquivo de metadados.
- `modelo/treinar_viirs.py`: separa períodos e áreas, compara modelos e calcula métricas.
- `modelo/prever_viirs.py`: reproduz consultas históricas da base auditada.
- `modelo/prever_atual.py`: consulta clima recente e devolve estimativas experimentais para os recortes disponíveis.
- `tests/test_modelo_viirs.py`: testa rótulos, vazamento temporal, separação, treinamento e consultas.

Para executar os testes:

```powershell
python -m pip install pytest
python -B -m pytest -q
```

### Próximas etapas para um serviço real

1. Acumular e avaliar previsões prospectivas antes de estabelecer desempenho operacional.
2. Obter e comparar previsões meteorológicas históricas emitidas antes de cada previsão de foco.
3. Avaliar novos períodos, biomas e regiões sem usar o mesmo teste para escolher o modelo.
4. Melhorar a revocação e entender a perda de cobertura por nuvens e o desempenho por região.
5. Só então ampliar a cobertura e integrar ao site, mantendo a distinção entre chance por área, observabilidade e probabilidade nacional.

### Credenciais

O `.env` local está incluído em `.gitignore` e não faz parte do histórico da `main` conferida. Uma branch local antiga contém um `.env` em seu histórico e não deve ser publicada ou mesclada sem remover as credenciais; a exclusão do arquivo atual não limpa commits antigos. Substitua as credenciais que possam ter sido expostas.
