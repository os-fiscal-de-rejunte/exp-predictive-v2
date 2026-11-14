# Teste do modelo brasileiro na Suíça

**Execução:** 14 de novembro de 2026.

**Dados avaliados:** 1º de janeiro a 31 de dezembro de 2025.

**Resultado:** o modelo não identificou nenhum dos 23 casos positivos observados no recorte e no período testados. A avaliação é exploratória e inconclusiva para medir o desempenho geral na Suíça.

## O que foi testado

O mesmo modelo treinado no recorte brasileiro do MATOPIBA foi aplicado à Suíça, sem retreinamento ou ajuste de limiar com dados suíços. O objetivo continuou sendo estimar a detecção, no próximo dia UTC, de pelo menos um pixel VIIRS com foco numa célula, **condicionada a uma máscara futura integralmente classificada**.

A grade foi criada independentemente dos focos, usando o limite suíço [Natural Earth](https://www.naturalearthdata.com/). Foram escolhidas 52 células de 0,25° × 0,25° inteiramente dentro do país. As regiões de fronteira e o restante do território ficaram fora; o teste não representa toda a Suíça.

Foram coletados os 365 grânulos disponíveis da máscara diária [VJ114A1.002 VIIRS/NOAA-20](https://doi.org/10.5067/VIIRS/VJ114A1.002), além de histórico de detecções [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/api/area/), clima [NASA POWER](https://power.larc.nasa.gov/docs/services/api/temporal/daily/), vegetação VIIRS VJ113A2.002 e relevo NASADEM. O histórico FIRMS usa produtos SP, enquanto a consulta atual do projeto usa NRT; o POWER histórico pode ter sido revisado. Isso não demonstra a disponibilidade exata desses valores em cada emissão histórica.

## Cobertura e resultados

| Item | Resultado |
| --- | ---: |
| Células/dia possíveis | 18.980 |
| Máscaras completas avaliáveis | 5.697 (30,02%) |
| Máscaras incompletas ou desconhecidas | 13.283 (69,98%) |
| Focos detectados com máscara completa | 23 |
| Detecções em máscaras incompletas | 4, preservadas fora do alvo condicionado |
| Casos classificados pelo modelo como foco | 0 |
| Casos positivos identificados | 0 de 23 |
| Revocação | 0,00% |
| Precisão | Não calculável: nenhum caso foi classificado como positivo |
| Acurácia | 99,60% |
| Brier, menor é melhor | 0,00624 |

Uma regra que dissesse sempre “sem detecção” obteria a **mesma acurácia de 99,60%** e revocação de 0%. Seu Brier seria 0,00404 neste conjunto. Portanto, a alta acurácia não indica que o modelo encontrou os focos, e suas probabilidades ficaram piores que essa referência no Brier deste teste.

Os 23 casos positivos ocorreram no modo `firms_sem_mascara`, usado em 5.396 células/dia. A maior probabilidade desse modo foi 23,80%, abaixo do limiar de 25,28% escolhido anteriormente no Brasil. Nas outras 301 células/dia, a ausência de vegetação utilizável levou ao modo `somente_clima`; não houve positivo observado nesse grupo. Alterar o limiar após ver este teste não demonstraria desempenho independente.

## Por que o teste tem limites

- Havia apenas 23 positivos, abaixo da meta de pelo menos 50; células e dias podem ser dependentes.
- A condição de máscara completa excluiu quase 70% das células/dia. Dias nublados ou sem classificação completa não viraram exemplos negativos.
- Na comparação das variáveis avaliáveis, todas as latitudes e longitudes suíças ficaram fora dos intervalos do treino brasileiro; 97,26% das temperaturas históricas agregadas e 45,83% das altitudes também ficaram fora dos respectivos intervalos.
- Foco de calor detectado não significa necessariamente incêndio, e não detecção não prova ausência de fogo.
- O estudo é retrospectivo e geográfico, não um teste prospectivo cego. As datas de 2025 coincidem parcialmente com o período usado para treinar o modelo no Brasil, embora as células suíças não tenham participado do treinamento.

O resultado mostra que o limiar e o modelo brasileiros não foram adequados para identificar os 23 casos deste teste suíço. Para avaliar uma versão própria para a região, seria necessário obter mais anos e regiões europeias, treinar e calibrar sem usar o teste reservado e depois validar em períodos e áreas independentes.

## Arquivos e reprodução

- `dados/teste_suica/relatorio_suica_2025.json`: métricas, cobertura, intervalos das entradas e hashes das fontes.
- `dados/teste_suica/previsoes_teste_suica_2025.csv`: previsão, limiar e resultado observado para cada célula/dia avaliável.
- `dados/teste_suica/grade_suica.csv`: geometria e identificação das 52 células.
- `dados/teste_suica/resumo_mascaras_suica_2025.json`: cobertura das máscaras e observações parciais.
- `modelo/suica.py` e `modelo/testar_suica.py`: coleta e avaliação reproduzíveis. As bases processadas do teste estão versionadas; dados brutos e cache ficam ignorados pelo Git.

Com as credenciais locais, o ambiente instalado e os arquivos do modelo regional disponíveis, a reprodução é feita na raiz do projeto:

```powershell
.\.venv\Scripts\python.exe -B -m modelo.suica grade
.\.venv\Scripts\python.exe -B -m modelo.suica mascaras
.\.venv\Scripts\python.exe -B -m modelo.suica firms
.\.venv\Scripts\python.exe -B -m modelo.suica clima
.\.venv\Scripts\python.exe -B -m modelo.suica geografia
.\.venv\Scripts\python.exe -B -m modelo.testar_suica mascaras
.\.venv\Scripts\python.exe -B -m modelo.testar_suica geografia
.\.venv\Scripts\python.exe -B -m modelo.testar_suica avaliar
```
