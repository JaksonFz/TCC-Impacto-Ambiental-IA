# Análise Comparativa da Eficiência Computacional e do Impacto Ambiental Estimado de Modelos Transformer na Inferência de Classificação de Sentimentos 

## Sobre o projeto

Este repositório contém os códigos e materiais desenvolvidos para o Trabalho de Conclusão de Curso (TCC), cujo objetivo é analisar comparativamente a eficiência computacional e o impacto ambiental estimado de modelos de Inteligência Artificial baseados na arquitetura Transformer.

O estudo concentra-se na tarefa de classificação de sentimentos (*Sentiment Analysis*), utilizando modelos pré-treinados para identificar sentimentos positivos e negativos em sentenças textuais.

A pesquisa busca comparar diferentes modelos considerando não apenas a qualidade das classificações, mas também os recursos computacionais necessários para executar suas inferências, com atenção especial ao tempo de processamento, consumo energético estimado e emissões de CO₂ equivalente.

## Modelos avaliados

Foram selecionados quatro modelos disponibilizados na plataforma Hugging Face:

| Modelo | Identificador |
|---|---|
| BERT Tiny | `takedarn/bert-tiny-sst2` |
| BERT Medium | `takedarn/bert-medium-sst2` |
| DistilBERT | `distilbert/distilbert-base-uncased-finetuned-sst-2-english` |
| BERT Base | `yoshitomo-matsubara/bert-base-uncased-sst2` |

## Tecnologias utilizadas

O projeto utiliza as seguintes ferramentas e bibliotecas:

- **Python:** linguagem utilizada no desenvolvimento dos experimentos.
- **PyTorch:** execução dos modelos e operações de inferência.
- **Hugging Face Transformers:** carregamento dos modelos pré-treinados e tokenizadores.
- **Hugging Face Datasets:** obtenção e manipulação do conjunto de dados SST-2.
- **CodeCarbon:** estimativa do consumo energético e das emissões de CO₂ equivalente.
- **Psutil:** monitoramento do uso de memória RAM pelo processo.
- **Pandas:** organização e exportação dos resultados.
- **Scikit-learn:** cálculo das métricas de avaliação da classificação.

## Metodologia experimental

Os experimentos são realizados utilizando a divisão de validação (*validation*) do conjunto de dados SST-2, composto por 872 sentenças em inglês.

Todos os modelos são executados em CPU, sem utilização de GPU, sob as mesmas configurações experimentais:

| Parâmetro | Configuração |
|---|---:|
| Dispositivo de processamento | CPU |
| Tamanho do lote (*Batch Size*) | 8 |
| Comprimento máximo (*Max Length*) | 128 tokens |
| Passagens pelo conjunto de dados | 10 |
| Repetições por modelo | 5 |
| Threads do PyTorch | 4 |

Cada repetição realiza dez passagens completas pelo conjunto de validação, totalizando 8.720 inferências por execução e 43.600 inferências por modelo ao longo das cinco repetições.

Antes de cada execução monitorada, é realizado um aquecimento (*warm-up*), que consiste em uma passagem completa pelo conjunto de dados. Essa etapa não é contabilizada nas métricas principais.

O tempo de processamento é medido com `time.perf_counter()`, enquanto o consumo energético e as emissões são estimados por meio do CodeCarbon. O monitoramento de memória utiliza o indicador RSS (*Resident Set Size*) disponibilizado pelo Psutil.

A avaliação da classificação considera quatro métricas: acurácia, precisão, revocação (*recall*) e F1-score. Essas métricas são calculadas sobre as 872 amostras únicas, utilizando as previsões da primeira passagem pelo conjunto de validação.

## Métricas coletadas

Durante os experimentos, são registrados indicadores de desempenho computacional, consumo de recursos e qualidade da classificação.

| Métrica | Descrição |
|---|---|
| Tempo de execução | Tempo necessário para realizar as dez passagens de inferência. |
| Inferências por segundo | Taxa de processamento calculada a partir do total de inferências e do tempo de execução. |
| Energia consumida | Consumo energético estimado pelo CodeCarbon, em kWh. |
| Emissões de CO₂ equivalente | Emissões estimadas pelo CodeCarbon, em kgCO₂eq. |
| Memória de pico | Maior consumo RSS observado pelo processo, em MB. |
| Acurácia | Proporção de classificações corretas. |
| Precisão | Proporção de previsões positivas que estão corretas. |
| Recall | Proporção de exemplos positivos corretamente identificados. |
| F1-score | Média harmônica entre precisão e recall. |

## Estrutura dos resultados

Durante a execução, o programa cria automaticamente o diretório `resultados_experimento_definitivo`, responsável por armazenar os arquivos gerados.

São produzidos três tipos de registros:

- **resultados_individuais.csv:** contém os resultados detalhados de cada repetição, permitindo analisar individualmente o comportamento dos modelos.
- **resumo_modelos.csv:** apresenta a média e o desvio-padrão das métricas agrupadas por modelo.
- **Arquivos emissions:** gerados pelo CodeCarbon, armazenam os registros individuais do monitoramento energético e das emissões estimadas de cada execução.

## Como executar

### 1. Clonar o repositório

```bash
git clone https://github.com/JaksonFz/TCC-Impacto-Ambiental-IA.git
cd TCC-Impacto-Ambiental-IA
```

### 2. Criar um ambiente virtual

```bash
python -m venv .venv
```

No Windows:

```bash
.venv\Scripts\activate
```

No Linux ou WSL2:

```bash
source .venv/bin/activate
```

### 3. Instalar as dependências

Caso o repositório disponibilize um arquivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Executar o experimento

```bash
python nome_do_script.py
```

Substitua `nome_do_script.py` pelo nome do arquivo principal do experimento.

Na primeira execução, é necessária conexão com a internet para obter os modelos e o conjunto de dados da Hugging Face, caso ainda não estejam disponíveis localmente.

## Considerações e limitações

O consumo energético e as emissões de CO₂ equivalente são estimativas produzidas pelo CodeCarbon, não medições físicas realizadas por equipamentos externos.

A medição principal considera a etapa de inferência, não incluindo o carregamento dos modelos, a tokenização e a preparação dos lotes. O aquecimento também ocorre antes do início do monitoramento.

O indicador de memória representa o consumo RSS do processo Python, incluindo bibliotecas e estruturas utilizadas durante a execução, e não somente a memória ocupada pelos parâmetros dos modelos.

O parâmetro WUE está configurado como zero no monitoramento do CodeCarbon, portanto, este experimento não realiza uma medição direta do consumo hídrico.
