# Relatório de evolução do modelo de hotspots

**Situação:** 14 de novembro de 2026.

**Classificação:** experimento regional, sem validação para alertas ou decisões operacionais.

**Projeto:** `HACKATON-NASA-2`.

## O que o modelo prevê

Para cada célula de 0,25° do recorte piloto do MATOPIBA, o modelo estima a probabilidade de o VIIRS/NOAA-20 registrar pelo menos um pixel de foco no dia UTC seguinte, **condicionada a todos os pixels da futura máscara estarem classificados**. Foco detectado não é sinônimo de incêndio confirmado; uma não detecção também não prova ausência de incêndio.

O recorte vai de 12°S a 8°S e de 47°O a 43°O e contém 256 células. Ele foi gerado independentemente dos focos e verificado contra os limites brasileiros Natural Earth. Não representa todo o MATOPIBA, o Brasil ou o mundo.

## Antes de começar

A cópia integral anterior está em `../HACKATON-NASA-2_BACKUP_20261114_011539/`. Foram conferidos 151 arquivos por SHA-256, sem diferenças. O backup contém arquivos locais sensíveis e deve permanecer fora do Git. O estado inicial e as modificações preexistentes estão registrados em [ESTADO_INICIAL_EVOLUCAO.md](ESTADO_INICIAL_EVOLUCAO.md). As tabelas originais do banco e os arquivos originais de `DATA` não foram alterados.

O piloto anterior usava oito recortes de 41 × 41 pixels. Sua avaliação tinha 131 exemplos e 14 positivos, com acurácia de 82,44% e revocação de 50%. A grade nova e o conjunto de teste mudaram: **não é correto comparar diretamente essas porcentagens como se medissem o mesmo problema**.

## Dados reunidos

| Fonte | Uso neste experimento |
| --- | --- |
| [VJ114A1.002 VIIRS/NOAA-20](https://doi.org/10.5067/VIIRS/VJ114A1.002) | Máscaras diárias nominais de 1 km para construir o alvo e parte do histórico. |
| [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/api/area/) | Histórico de registros MODIS e VIIRS; produtos históricos SP e consultas recentes NRT são identificados separadamente. |
| [NASA POWER Daily](https://power.larc.nasa.gov/docs/services/api/temporal/daily/) | Temperatura, umidade relativa, precipitação, vento e umidade do solo históricos. |
| VIIRS VJ113A2.002 e NASADEM_HGT.001, obtidos pelo [NASA Earthdata](https://search.earthdata.nasa.gov/) | NDVI, EVI e altitude, com controle de qualidade e data de disponibilidade dos compostos. |
| [MCD12Q1.061 MODIS](https://doi.org/10.5067/MODIS/MCD12Q1.061) | Cobertura anual do solo, usada como contexto e em um experimento separado. |
| [Natural Earth](https://www.naturalearthdata.com/) | Limite brasileiro usado na grade; o arquivo e seu hash constam da proveniência local. |

Entre 1º de janeiro de 2024 e 31 de dezembro de 2025, foram obtidos 1.455 dos 1.462 grânulos diários esperados para os dois tiles. Sete grânulos não estavam no catálogo. A série tem 187.136 células/dia:

| Situação | Células/dia |
| --- | ---: |
| Máscara completa e foco detectado | 9.989 |
| Máscara completa sem detecção | 117.454 |
| Máscara incompleta ou desconhecida | 59.693 |
| Dessas incompletas, com evidência de foco | 1.081 |

Uma máscara incompleta continua desconhecida para o alvo condicionado, mesmo quando há evidência parcial de foco. Essa evidência não foi transformada em negativo e pode compor o histórico quando a publicação é anterior à emissão.

Também foram obtidos 169.777 registros FIRMS históricos na janela de coleta e clima para 63 células meteorológicas. Uma célula de clima pode alimentar várias células da grade de previsão. Para vegetação, foram extraídos 25 compostos históricos mensais selecionados e exigida qualidade mínima; a altitude cobre as 256 células. A cobertura do solo anual de 2022 a 2025 foi agregada em 1.024 células/ano; o uso considera somente pixels terrestres classificados, exige pelo menos 20% de cobertura válida e respeita a data de produção e atualização no catálogo. Valores indisponíveis permanecem ausentes.

A cobertura varia com a estação: em dezembro de 2025 apenas 1.561 de 7.936 células/dia, ou 19,67%, tinham máscara completa. Isso significa que a avaliação é especialmente seletiva em alguns meses. A cobertura mensal e as faixas de calibração estão em `modelo/artefatos/evolucao_grade/auditoria_grade.json`.

## Treinamento e avaliação

O treino usou 48.155 exemplos de 160 células; calibração e seleção do limiar tiveram períodos posteriores separados. O teste temporal e espacial, de setembro a dezembro de 2025, contém 16.000 células/dia com máscaras completas, sendo 2.689 positivas. Destas, 5.006 células/dia, com 533 positivas, pertencem às 64 células reservadas espacialmente. Duas colunas de células foram excluídas entre ajuste e teste espacial, formando afastamento aproximado de 55 km.

Foram comparadas regressão logística e árvores de gradiente XGBoost em dobras cronológicas. A calibração e o limiar foram decididos antes do período de teste. Os históricos de máscara só entram quando a data de produção permite; clima usa uma janela histórica de D−10 a D−3, e a vegetação só entra após o fim e a publicação do composto. Isso não prova que o clima revisado e o FIRMS SP estivessem disponíveis exatamente daquele jeito nas emissões históricas.

| Medida no teste | Modo com histórico de máscara | Modo atual sem máscara recente |
| --- | ---: | ---: |
| Acurácia | 74,12% | **77,99%** |
| Precisão | 33,50% | **36,68%** |
| Revocação | 54,82% | **42,62%** |
| Casos positivos encontrados | 1.474 / 2.689 | 1.146 / 2.689 |
| Average precision | 0,3811 | 0,3594 |
| Brier, menor é melhor | 0,1232 | 0,1250 |

O limiar de cada modelo foi escolhido no período de decisão para buscar maior revocação com precisão de pelo menos 30%. Uma referência sazonal testada nas mesmas células e datas teve Brier 0,1383. O modo atual reduziu esse erro em aproximadamente 9,6%; o modo com máscara, em 11,0%. A meta inicial também exigia revocação de pelo menos 60% e um teste novo: ela **não foi atingida nem demonstrada**.

No teste espacial, o modo atual encontrou 154 dos 533 positivos: revocação de **28,89%**, precisão de **34,07%** e Brier 0,0868. É um resultado fraco para a área reservada e continua dentro da mesma região, não é validação em outro bioma.

Uma regra que classificasse todas as células como sem detecção teria acurácia de 83,19%, mas encontraria 0% dos positivos. Acurácia isolada seria enganosa. O intervalo exploratório de 95% da revocação do modo atual foi de 37,10% a 48,34%, obtido por 300 reamostragens de dias inteiros; ele não elimina a dependência entre dias e regiões.

As faixas de calibração foram recalculadas a partir das previsões do teste. Para o modo atual, por exemplo, a faixa de 10% a 20% teve probabilidade média de 14,23% e frequência observada de 17,27%, em 4.452 células/dia. Isso descreve esse teste, sem garantir calibração futura. As linhas de uma mesma data ou área não são eventos independentes.

Os testes sem vegetação, sem terreno e sem outros grupos de variáveis foram registrados separadamente. Neste conjunto, adicionar vegetação e terreno não demonstrou melhora consistente; a comparação não prova causalidade.

Foram testados também vizinhos, desvios climáticos sazonais calculados dentro de cada partição de ajuste e uma floresta aleatória. No modo atual, o candidato com vizinhança teve precisão de 37,03% e revocação de 42,17% no teste geral, comparados a 36,68% e 42,62% do modelo principal; no teste espacial, a revocação foi 28,33%, contra 28,89%. Não houve evidência consistente para substituir o modelo principal. Os resultados estão em `modelo/artefatos/evolucao_vizinhanca/relatorio_vizinhanca.json`.

Na comparação com cobertura do solo, o candidato sem máscara recente teve precisão geral de 39,61% e revocação de 34,66%; no teste espacial, encontrou apenas 18,76% dos positivos, contra 28,89% do modelo principal. Por isso, a cobertura permanece como contexto na consulta, sem entrar na previsão principal. O relatório `modelo/artefatos/evolucao_cobertura_solo/relatorio_cobertura_solo.json` também separa o desempenho por classe dominante: para floresta havia apenas 32 positivos, insuficientes para conclusão. O guia do produto aponta possíveis erros de classificação, inclusive sub-representação de agricultura em algumas áreas tropicais.

Explicações individuais e uma referência histórica específica por célula ainda não estão disponíveis como saídas validadas.

## Previsão atual e uso no futuro site

A execução de 14/11/2026 gerou uma previsão para 15/11/2026 UTC para as 256 células. As máscaras padrão mais recentes disponíveis no momento eram de 06/11, então as 256 usaram o modo treinado separadamente **sem máscara recente**. A consulta traz FIRMS NRT, clima histórico, NDVI/EVI, altitude, cobertura do solo, datas, origem, contagens históricas e soma de FRP quando disponível. A cobertura de 2025 foi publicada em 2026 e descreve o ano de referência, não uma medição atual. A soma de FRP reúne valores de potência das detecções; não é uma medida de energia queimada nem uma explicação causal.

Arquivos principais:

- `modelo/artefatos/evolucao_grade/previsao_atual.json`: resultados por célula e avaliação retrospectiva do modo usado.
- `modelo/artefatos/evolucao_grade/previsao_atual.geojson`: geometrias e probabilidades para um futuro mapa.
- `modelo/artefatos/evolucao_grade/relatorio_grade.json` e `previsoes_teste.csv`: métricas e previsões do teste histórico.
- `modelo/artefatos/evolucao_grade/auditoria_grade.json`: recálculo das métricas, calibração em faixas e cobertura mensal.
- `modelo/artefatos/evolucao_vizinhanca/relatorio_vizinhanca.json` e `modelo/artefatos/evolucao_cobertura_solo/relatorio_cobertura_solo.json`: comparações exploratórias adicionais.
- `modelo/artefatos/evolucao_grade/previsoes_prospectivas/`: emissões guardadas antes do dia previsto.
- `modelo/artefatos/evolucao_grade/avaliacao_prospectiva.json`: situação da comparação com observações posteriores.

Todos os resultados são `experimental_nao_operacional`. A saída deixa nulas a probabilidade nacional e a probabilidade incondicional de incêndio. Quando falta clima essencial para uma célula, ela recebe `dados_insuficientes`, sem probabilidade. As fontes, datas de coleta, modelo e entradas têm hashes registrados. A previsão foi emitida sob demanda, não automaticamente às 23h UTC.

## Atualizar e validar

No PowerShell, dentro de `HACKATON-NASA-2`:

```powershell
.\.venv\Scripts\python.exe -B -m modelo.atualizar_grade
.\.venv\Scripts\python.exe -B -m modelo.auditar_grade
.\.venv\Scripts\python.exe -B -m modelo.avaliar_prospectivo --atualizar-observacoes
```

Para instalar as dependências em outro ambiente, consulte o [README](README.md). A atualização precisa de `EARTHDATA_TOKEN` e `FIRMS_MAP_KEY` no `.env` local, além da grade e do modelo treinado. As bases históricas processadas, o modelo e os relatórios necessários para a avaliação estão versionados. Grânulos brutos, cache, consultas atuais e registros prospectivos permanecem ignorados pelo Git. Joblib só deve ser carregado de fonte confiável.

A atualização diária reutiliza a cobertura anual do solo já processada. Para consultar e processar um ano publicado, por exemplo 2025, execute os comandos abaixo e depois atualize a previsão; altere os anos apenas após conferir a disponibilidade do produto:

```powershell
.\.venv\Scripts\python.exe -B -m modelo.coletar_cobertura_solo --inicio 2025 --fim 2025
.\.venv\Scripts\python.exe -B -m modelo.cobertura_solo
.\.venv\Scripts\python.exe -B -m modelo.atualizar_grade
```

A primeira previsão prospectiva foi registrada para 15/11/2026 e aguarda o fim do dia e a publicação das máscaras. O script escolhe a última emissão anterior ao início do dia, exclui máscaras incompletas do alvo e marca avaliações com menos de 50 positivos como inconclusivas. O SHA-256 detecta alterações locais, mas não é um carimbo de tempo certificado. A etapa prospectiva precisa acumular novas datas e não foi antecipada artificialmente.

## Limitações e trabalho restante

- Os dados de 2024–2025 já haviam sido examinados durante o desenvolvimento; o teste é retrospectivo e exploratório, não cego nem prospectivo.
- As entradas POWER históricas podem ter sido revisadas. As detecções FIRMS SP do treino podem diferir das NRT disponíveis na previsão. Uma janela conservadora reduz o risco temporal, mas não o elimina.
- O [ECMWF Open Data](https://www.ecmwf.int/en/forecasts/datasets/open-data) mantém no portal principal uma janela recente de rodadas; a [réplica AWS](https://registry.opendata.aws/ecmwf-forecasts/) pode manter dados mais antigos. Não foi montada nem verificada uma série histórica consistente de previsões emitidas para este experimento; por isso, previsão meteorológica do dia-alvo ficou fora do modelo.
- Permanecem necessários testes prospectivos com dados suficientes, novas regiões e biomas, avaliação de viés por disponibilidade e nuvens, e melhora da revocação espacial. A cobertura atual não autoriza generalização nacional ou mundial.
- Não há cálculo validado de probabilidade nacional, ignição, propagação, área queimada ou intervalo individual de incerteza. Não há site nem alerta operacional nesta entrega.
