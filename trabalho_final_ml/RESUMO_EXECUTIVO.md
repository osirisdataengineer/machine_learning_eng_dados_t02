# Resumo Executivo - Projeto 2: Risco de Queimadas

## 1. Objetivo do Projeto

Este projeto desenvolve uma solucao de Machine Learning para prever a ocorrencia de incendios florestais a partir de dados meteorologicos, espaciais e temporais. O problema foi modelado como uma tarefa de **classificacao binaria**, em que o modelo deve indicar se houve ou nao incendio.

A base utilizada foi o dataset **Forest Fires**, consumido localmente a partir do arquivo:

```text
dataset/forestfires.csv
```

A meta tecnica do Projeto 2 e obter **F1-Score superior a 0.65 para a classe positiva**, que representa ocorrencia de incendio.

## 2. Definicao da Variavel Alvo

O dataset original possui a coluna `area`, que informa a area queimada em hectares. Para adaptar o problema para classificacao binaria, foi criada a coluna `incendio`:

- `incendio = 1`: houve incendio, quando `area > 0`.
- `incendio = 0`: nao houve incendio, quando `area == 0`.

A coluna `area` foi removida das variaveis explicativas apos a criacao do alvo, pois sua permanencia no conjunto de features causaria **data leakage**. Em um cenario real de predicao, a area queimada ainda nao seria conhecida no momento da inferencia.

## 3. Analise e Preparacao dos Dados

Durante a analise exploratoria foram avaliados:

- Dimensao do conjunto de dados.
- Tipos das variaveis.
- Estatisticas descritivas.
- Valores ausentes.
- Distribuicao da variavel alvo.
- Possivel desbalanceamento entre classes.

As variaveis originais incluem atributos espaciais (`X`, `Y`), temporais (`month`, `day`), meteorologicos (`temp`, `RH`, `wind`, `rain`) e indices do sistema Fire Weather Index (`FFMC`, `DMC`, `DC`, `ISI`).

Foram consideradas hipoteses como a relacao entre maior temperatura, menor umidade relativa, meses mais secos e aumento do risco de incendio. Tambem foi considerada a importancia dos indices FWI, que representam condicoes associadas a combustivel seco e propagacao do fogo.

## 4. Engenharia de Atributos

Alem das variaveis originais, foram criadas features derivadas para enriquecer a representacao dos dados:

- `month_num`: conversao do mes textual para numero.
- `is_weekend`: indicador para sabado ou domingo.
- `temp_rh_ratio`: relacao entre temperatura e umidade relativa.
- `dry_wind_index`: combinacao entre vento, temperatura e umidade.
- `fwi_mean`: media dos indices `FFMC`, `DMC`, `DC` e `ISI`.

Essas transformacoes foram feitas linha a linha, sem uso de estatisticas globais da base e sem acesso a variavel alvo. Portanto, nao introduzem vazamento de dados.

## 5. Estrutura do Pipeline

O projeto utiliza `Pipeline` e `ColumnTransformer` do scikit-learn para garantir uma arquitetura reprodutivel e adequada para producao.

O pre-processamento foi dividido em duas partes:

- Variaveis numericas: imputacao pela mediana e padronizacao com `StandardScaler`.
- Variaveis categoricas: imputacao pela moda e codificacao com `OneHotEncoder(handle_unknown="ignore")`.

Todas as transformacoes ficam encapsuladas dentro do pipeline, de modo que os parametros de imputacao, escala e codificacao sejam aprendidos apenas no conjunto de treino. Isso evita vazamento de informacao do conjunto de teste.

A divisao treino/teste foi feita com `train_test_split`, `random_state=42` e `stratify=y`, mantendo a proporcao das classes nos dois conjuntos.

## 6. Modelagem e Otimizacao

Foram comparados diferentes algoritmos de classificacao:

- `LogisticRegression`
- `RandomForestClassifier`
- `ExtraTreesClassifier`
- `GradientBoostingClassifier`
- `AdaBoostClassifier`

A busca de hiperparametros foi realizada com `GridSearchCV`, usando validacao cruzada estratificada e `scoring="f1"`. Os hiperparametros foram configurados com o prefixo correto da etapa do pipeline, como `model__C`, `model__n_estimators`, `model__max_depth` e `model__learning_rate`.

Como o objetivo principal e maximizar o F1-Score da classe positiva, tambem foi realizado ajuste do threshold de decisao. O threshold foi escolhido usando apenas o conjunto de treino por meio de validacao cruzada, preservando o conjunto de teste exclusivamente para a avaliacao final.

## 7. Resultado Final

O melhor pipeline selecionado utilizou:

```text
Modelo: LogisticRegression
Threshold final: 0.27
Melhor parametro: model__C = 0.01
```

Na avaliacao final sobre o conjunto de teste, o modelo obteve:

```text
F1-Score da classe positiva: 0.6879
```

Com esse resultado, o projeto atinge a meta tecnica definida para o Projeto 2:

```text
F1-Score > 0.65
```

O resultado indica que o pipeline conseguiu priorizar corretamente a deteccao da classe positiva, que representa ocorrencia de incendio. Esse foco e coerente com o contexto do problema, pois falhar na identificacao de um possivel incendio pode ter impacto operacional relevante.

## 8. Serializacao e Simulacao de Producao

O pipeline final foi serializado com `joblib`, gerando o artefato:

```text
notebook/pipeline_queimadas_otimizado.joblib
```

O artefato salvo contem:

- Pipeline completo de pre-processamento e modelo.
- Threshold final de decisao.
- Lista de features esperadas na inferencia.
- Nome do modelo selecionado.

Tambem foi implementada a funcao:

```python
predizer_incendio(payload)
```

Essa funcao simula uma inferencia em producao. Ela recebe um dicionario Python com dados brutos, recria as features derivadas, carrega o artefato `.joblib`, aplica o pipeline treinado e retorna a predicao final como `0` ou `1`.

O payload de teste inclui valor nulo simulado, demonstrando que o pipeline e capaz de lidar com dados faltantes por meio dos imputadores definidos no pre-processamento.

## 9. Conclusao

O projeto entrega uma solucao completa e reprodutivel para classificacao de risco de queimadas. A implementacao segue boas praticas de engenharia de Machine Learning, incluindo separacao correta entre treino e teste, prevencao de data leakage, uso de `Pipeline` e `ColumnTransformer`, comparacao de modelos, otimizacao de hiperparametros, ajuste de threshold, avaliacao no conjunto de teste e serializacao do pipeline para uso em producao.

O modelo final atingiu a meta estabelecida, obtendo **F1-Score de 0.6879 para a classe positiva**. Dessa forma, a solucao atende aos requisitos tecnicos do Projeto 2 e fornece uma base consistente para evolucoes futuras, como monitoramento em producao, ampliacao da busca de hiperparametros ou testes com modelos adicionais.
