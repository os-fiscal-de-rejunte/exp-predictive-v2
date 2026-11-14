# Modelo experimental de detecção de focos

Este projeto contém um piloto treinado com dados reais para estimar a detecção de pelo menos um pixel de foco em uma área, no próximo dia UTC. A estimativa é **condicionada a todos os pixels da futura máscara VIIRS estarem classificados**. Não é a probabilidade de incêndio, de ignição ou de haver algum foco no país inteiro.

O piloto cobre oito recortes selecionados no Brasil, cada um com 41 × 41 pixels de aproximadamente 927 m em projeção sinusoidal. As áreas não foram sorteadas para representar o Brasil e algumas podem abranger regiões próximas a fronteiras. A consulta por país devolve somente esses recortes, não um mapa nacional ou mundial. O projeto ainda não é apropriado para alertas ou decisões operacionais.

## Dados e rótulo

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

## Como executar

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

**Limite da consulta atual:** ela ainda não coleta o histórico recente MODIS/VIIRS. Esses campos entram como ausentes e são tratados pelo imputador treinado, com indicadores de ausência. A saída é experimental, não libera uma probabilidade incondicional e não deve ser usada como alerta. O teste de sensibilidade a essa ausência está descrito abaixo.

## Como o modelo foi avaliado

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

## Organização

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

## Próximas etapas para um serviço real

1. Construir uma grade terrestre independente dos focos, cobrindo as regiões e datas de interesse, e obter máscaras e detecções com versões e proveniência conhecidas para períodos contínuos.
2. Integrar dados recentes de MODIS e VIIRS por recorte, preservando falhas, latência, geometria e ausência de observação. A API FIRMS limita cada consulta por área a cinco dias e usa `FIRMS_MAP_KEY`; [consulte o contrato da NASA](https://firms.modaps.eosdis.nasa.gov/api/area/).
3. Obter vegetação, terreno e histórico de previsões meteorológicas realmente disponíveis antes da emissão; validar resolução, unidades e datas de publicação.
4. Reavaliar com anos e áreas que não tenham sido usados para escolher locais ou modelos, com amostragem representativa, análise de calibração, limites de confiança e critérios de liberação definidos antes do teste.
5. Só então ampliar para países e integrar ao site, mantendo a distinção entre chance por área, observabilidade e probabilidade agregada nacional.

## Credenciais

O `.env` local foi retirado do índice e incluído em `.gitignore`; seu conteúdo permanece no computador. Ele já constava em um commit anterior: isso não apaga o histórico. Revise o acesso ao repositório e substitua as credenciais que possam ter sido expostas. A limpeza do histórico compartilhado exige coordenação, pois reescreve commits.
