# Resumo Executivo - Projeto 2: Risco de Queimadas

## 1. Visao Geral do Projeto

Este projeto tem como objetivo construir um pipeline de Machine Learning para prever a ocorrencia de incendios florestais a partir de dados meteorologicos, espaciais e temporais. A base utilizada foi o dataset **Forest Fires**, disponibilizado no arquivo local `dataset/forestfires.csv`.

O problema foi tratado como uma tarefa de **classificacao binaria**. A coluna original `area`, que representa a area queimada em hectares, foi transformada na variavel alvo `incendio`:

- `incendio = 1`: houve incendio, quando `area > 0`.
- `incendio = 0`: nao houve incendio, quando `area == 0`.

A meta tecnica definida para o Projeto 2 e obter **F1-Score superior a 0.65 para a classe positiva**, isto e, para a classe que representa ocorrencia de incendio.

## 2. Entendimento dos Dados

O conjunto de dados possui variaveis espaciais, temporais e meteorologicas. Entre as principais variaveis explicativas estao:

- `X` e `Y`: coordenadas espaciais da regiao monitorada.
- `month` e `day`: mes e dia da semana da observacao.
- `FFMC`, `DMC`, `DC` e `ISI`: indices relacionados ao sistema Fire Weather Index.
- `temp`: temperatura.
- `RH`: umidade relativa.
- `wind`: velocidade do vento.
- `rain`: volume de chuva.

Durante a analise exploratoria, foram avaliados o formato da base, os tipos das colunas, estatisticas descritivas, valores ausentes e distribuicao da variavel alvo. Essa etapa foi importante para identificar o comportamento das variaveis, verificar possivel desbalanceamento entre classes e orientar a escolha das transformacoes aplicadas no pipeline.

Algumas hipoteses consideradas foram:

- Temperaturas mais altas e umidade relativa mais baixa podem estar associadas a maior risco de incendio.
- Meses mais secos ou quentes podem concentrar maior ocorrencia de queimadas.
- Os indices do sistema FWI podem carregar informacoes relevantes sobre combustivel seco e condicoes de propagacao do fogo.
- A velocidade do vento pode influenciar a propagacao, embora isoladamente nao determine a ocorrencia do incendio.

## 3. Estrategia de Engenharia de Dados

Para evitar **data leakage**, a coluna `area` foi removida das variaveis explicativas depois da criacao do alvo `incendio`. Essa decisao e essencial, pois `area` indica diretamente se houve ou nao queimada e, se usada no treinamento, faria o modelo aprender uma informacao que nao estaria disponivel em uma situacao real de predicao.

A divisao entre treino e teste foi feita imediatamente apos a definicao de `X` e `y`, usando `train_test_split` com `random_state=42` e `stratify=y`. O uso de `stratify` preserva a proporcao das classes nos conjuntos de treino e teste, tornando a avaliacao mais fiel.

O pre-processamento foi implementado com `ColumnTransformer`, separando variaveis numericas e categoricas:

- Variaveis numericas: imputacao pela mediana e padronizacao com `StandardScaler`.
- Variaveis categoricas: imputacao pela moda e codificacao com `OneHotEncoder(handle_unknown="ignore")`.

Todas essas transformacoes foram encapsuladas em pipelines do scikit-learn. Isso garante que os parametros de imputacao, escala e codificacao sejam aprendidos apenas no conjunto de treino durante o `fit`, evitando vazamento de informacao do conjunto de teste.

## 4. Modelagem e Otimizacao

Foram comparados tres algoritmos de classificacao:

- `LogisticRegression`
- `RandomForestClassifier`
- `GradientBoostingClassifier`

A comparacao entre modelos atende ao requisito de experimentar pelo menos dois algoritmos diferentes. Para a sintonia de hiperparametros, foi utilizado `GridSearchCV` com validacao cruzada estratificada (`StratifiedKFold`) e metrica de otimizacao `f1`.

Os parametros dos modelos foram configurados com o prefixo correto da etapa do pipeline, como `model__n_estimators`, `model__max_depth` e `model__learning_rate`. Essa abordagem permite que a busca de hiperparametros avalie o pipeline completo, incluindo pre-processamento e estimador, de forma consistente.

O modelo com melhor desempenho no conjunto de teste foi o **Gradient Boosting**, com os seguintes hiperparametros selecionados:

```python
{
    "model__learning_rate": 0.03,
    "model__max_depth": 2,
    "model__min_samples_leaf": 5,
    "model__n_estimators": 50
}
```

## 5. Resultados Obtidos

A avaliacao final foi realizada exclusivamente no conjunto de teste, preservado desde o inicio do processo. As metricas analisadas incluíram matriz de confusao, precision, recall e F1-Score por classe.

O melhor modelo obteve:

```text
F1-Score da classe positiva: 0.6290
```

Embora o resultado tenha ficado proximo da meta definida para o projeto, ele ainda nao ultrapassou o limiar exigido de **0.65**. Isso indica que a solucao esta tecnicamente correta em termos de arquitetura, pipeline e validacao, mas ainda possui margem para melhoria na etapa de desempenho preditivo.

Possiveis caminhos para melhorar o resultado incluem:

- Testar tratamento de desbalanceamento, como ajuste de threshold ou tecnicas com `imblearn`.
- Avaliar modelos adicionais, como `ExtraTreesClassifier`, `XGBoost`, `LightGBM` ou `CatBoost`.
- Criar novas features a partir de variaveis temporais ou meteorologicas.
- Realizar busca de hiperparametros mais ampla.
- Avaliar metricas complementares, como precision, recall e curva precision-recall.

## 6. Serializacao e Simulacao de Producao

O pipeline final foi serializado com `joblib`, gerando o arquivo:

```text
notebook/pipeline_queimadas.joblib
```

Esse arquivo contem o fluxo completo de inferencia, incluindo:

- Pre-processamento numerico.
- Pre-processamento categorico.
- Modelo final ajustado.

Tambem foi implementada a funcao `predizer_incendio(payload)`, que simula o comportamento de uma API de producao. A funcao recebe um dicionario Python com dados brutos, converte o payload em um `DataFrame`, carrega o pipeline salvo em disco e retorna a predicao final como `0` ou `1`.

O payload de teste inclui valor nulo simulado, demonstrando que o pipeline e capaz de lidar com dados faltantes por meio dos imputadores definidos na etapa de pre-processamento.

## 7. Conclusao

O projeto implementa uma solucao completa de Machine Learning em formato reprodutivel e orientado a producao. A arquitetura segue boas praticas de MLOps inicial, com separacao correta entre treino e teste, prevencao de data leakage, uso de `Pipeline` e `ColumnTransformer`, otimizacao de hiperparametros, comparacao entre modelos, serializacao do pipeline e simulacao de inferencia com payload bruto.

O principal ponto de atencao e o desempenho final, pois o F1-Score da classe positiva ficou em **0.6290**, abaixo da meta de **0.65**. Ainda assim, a estrutura tecnica entregue permite evolucoes claras e controladas para melhorar a performance, mantendo a robustez e a reprodutibilidade da solucao.
