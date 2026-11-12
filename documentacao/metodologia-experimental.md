Documentação do experimento de inferência e impacto ambiental
1. Visão geral
Este projeto tem como objetivo avaliar o desempenho computacional e o impacto ambiental estimado de diferentes modelos de inteligência artificial baseados na arquitetura Transformer, utilizando uma tarefa de classificação de sentimentos.

O experimento compara quatro modelos: BERT Tiny, BERT Medium, DistilBERT e BERT Base. Para isso, são observados indicadores como tempo de execução, taxa de processamento, consumo energético estimado, emissões de CO₂ equivalente, utilização de memória e métricas de classificação.

As execuções são realizadas localmente, utilizando exclusivamente a CPU, com o auxílio das bibliotecas PyTorch, Hugging Face Transformers e CodeCarbon.
2. Bibliotecas e dependências
O projeto utiliza bibliotecas Python para executar os modelos, manipular os dados e monitorar os recursos computacionais.
import osimport gcimport timeimport torchimport psutilimport pandas as pdfrom datasets import load_datasetfrom transformers import AutoTokenizer, AutoModelForSequenceClassificationfrom codecarbon import EmissionsTrackerfrom sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



As bibliotecas torch e transformers são responsáveis pelo carregamento e processamento dos modelos. A biblioteca datasets permite acessar o conjunto de dados SST-2, enquanto codecarbon realiza as estimativas de consumo energético e emissões.

Já psutil permite acompanhar a utilização de memória pelo processo, pandas organiza os resultados e sklearn.metrics fornece as métricas utilizadas na avaliação dos modelos. As bibliotecas os, gc e time auxiliam no gerenciamento de arquivos, liberação de memória e medição do tempo de execução.
3. Configuração experimental
Os modelos utilizados são definidos em um dicionário, que associa seus nomes aos respectivos identificadores disponíveis no Hugging Face.
MODELOS = {    "BERT Tiny": "takedarn/bert-tiny-sst2",    "BERT Medium": "takedarn/bert-medium-sst2",    "DistilBERT": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",    "BERT Base": "yoshitomo-matsubara/bert-base-uncased-sst2"}



Essa organização facilita a inclusão ou substituição de modelos, permitindo que o mesmo procedimento experimental seja aplicado a todos eles.

Os principais parâmetros utilizados durante o experimento são apresentados a seguir:
BATCH_SIZE = 8MAX_LENGTH = 128PASSAGENS = 10REPETICOES = 5FATOR_CARBONO = 98.348



O parâmetro BATCH_SIZE estabelece que oito textos serão processados simultaneamente, enquanto MAX_LENGTH limita cada sequência a 128 tokens. O parâmetro PASSAGENS determina que o conjunto de dados será processado dez vezes por repetição, e REPETICOES estabelece cinco execuções independentes para cada modelo.

O FATOR_CARBONO representa a intensidade de carbono utilizada pelo CodeCarbon, expressa em gramas de CO₂ equivalente por quilowatt-hora (gCO₂eq/kWh).
4. Organização dos arquivos de resultados
O experimento utiliza um diretório específico para armazenar os arquivos gerados durante sua execução.
PASTA_RESULTADOS = "resultados_experimento_definitivo"ARQUIVO_RESULTADOS = os.path.join(    PASTA_RESULTADOS, "resultados_individuais.csv")ARQUIVO_RESUMO = os.path.join(    PASTA_RESULTADOS, "resumo_modelos.csv")os.makedirs(PASTA_RESULTADOS, exist_ok=True)



O diretório é criado automaticamente caso ainda não exista. O arquivo resultados_individuais.csv armazena os dados de cada repetição, enquanto resumo_modelos.csv apresenta os resultados consolidados por modelo.

Antes do início do experimento, o arquivo de resultados individuais existente é removido, evitando que informações de execuções anteriores sejam incorporadas aos novos resultados.
5. Preparação do ambiente de execução
Para padronizar o processamento, são estabelecidos limites de utilização de threads do PyTorch.
torch.set_num_threads(4)torch.set_num_interop_threads(4)processo = psutil.Process(os.getpid())



As configurações estabelecem quatro threads para operações computacionais e quatro threads para operações paralelas. Em seguida, psutil.Process identifica o processo atual, permitindo acompanhar sua utilização de memória durante o experimento.

O conjunto de dados é carregado utilizando a divisão de validação do SST-2:
dataset = load_dataset("stanfordnlp/sst2")["validation"]textos = dataset["sentence"]rotulos_reais = dataset["label"]



São extraídos os textos e seus respectivos rótulos, que posteriormente serão utilizados na classificação e avaliação dos modelos. A divisão de validação utilizada contém 872 amostras.
6. Carregamento e preparação dos modelos
O processamento é realizado sequencialmente, percorrendo os modelos definidos anteriormente.
tokenizer = AutoTokenizer.from_pretrained(identificador)modelo = AutoModelForSequenceClassification.from_pretrained(    identificador)modelo.to("cpu")modelo.eval()



O tokenizador e o modelo são carregados a partir dos identificadores do Hugging Face. A instrução modelo.to("cpu") direciona a execução para o processador, enquanto modelo.eval() configura o modelo para o modo de avaliação, desativando comportamentos específicos do treinamento, como o dropout.

Após o carregamento, os textos são divididos em lotes e convertidos em tensores:
for inicio in range(0, len(textos), BATCH_SIZE):    entradas = tokenizer(        textos[inicio:inicio + BATCH_SIZE],        return_tensors="pt",        padding="max_length",        truncation=True,        max_length=MAX_LENGTH    )    lotes.append(entradas)



A tokenização transforma os textos em representações numéricas que podem ser processadas pelo modelo. O preenchimento padroniza o comprimento das sequências, enquanto o truncamento limita textos que ultrapassam o tamanho máximo estabelecido.

Os lotes são armazenados na lista lotes para serem reutilizados nas repetições seguintes.
7. Aquecimento e execução experimental
Antes de iniciar cada repetição, é realizada uma execução preliminar dos lotes:
with torch.inference_mode():    for lote in lotes:        modelo(**lote)



Essa etapa, conhecida como warm-up, permite executar previamente as operações do modelo, reduzindo possíveis interferências associadas à inicialização do processamento.

Após o aquecimento, inicia-se a configuração do monitoramento ambiental.
7.1. Monitoramento ambiental
O CodeCarbon é utilizado para estimar o consumo energético e as emissões de CO₂ equivalente durante o processamento.
tracker = EmissionsTracker(    project_name=f"TCC_{nome.replace(' ', '_')}_R{repeticao}",    measure_power_secs=1,    tracking_mode="process",    pue=1.0,    wue=0,    api_call_interval=-1,    force_carbon_intensity_g_co2e_kwh=FATOR_CARBONO,    output_dir=PASTA_RESULTADOS,    output_file=nome_arquivo)tracker.start()



Cada repetição recebe uma identificação própria, permitindo diferenciar os arquivos de monitoramento gerados pelos modelos.

O parâmetro measure_power_secs estabelece o intervalo de medição de potência, enquanto tracking_mode="process" configura o monitoramento do processo. A intensidade de carbono é fixada conforme o valor definido anteriormente.

Os parâmetros PUE e WUE são mantidos em 1,0 e 0, respectivamente, não representando medições reais de infraestrutura de data centers.
7.2. Processamento das inferências
Com o monitoramento iniciado, o código executa dez passagens completas pelo conjunto de dados.
with torch.inference_mode():    for passagem in range(PASSAGENS):        for lote in lotes:            saida = modelo(**lote)            if passagem == 0:                predicoes.extend(                    torch.argmax(                        saida.logits,                        dim=1                    ).cpu().tolist()                )



A função torch.inference_mode() reduz operações desnecessárias relacionadas ao acompanhamento de gradientes, tornando o processamento mais adequado para inferências.

As classificações são armazenadas somente durante a primeira passagem. Nas demais, os lotes continuam sendo processados para compor o tempo e o consumo energético monitorados, sem duplicar as previsões utilizadas na avaliação.

Ao término do processamento, o tempo total é calculado e o monitoramento do CodeCarbon é encerrado.
8. Monitoramento da memória
Durante as inferências, o consumo de memória é acompanhado por meio do psutil.
memoria_atual = (    processo.memory_info().rss / (1024 ** 2))memoria_pico = max(    memoria_pico,    memoria_atual)



A propriedade rss representa a memória residente do processo, ou seja, a quantidade de memória física ocupada por ele.

A conversão para megabytes é realizada dividindo o valor por \(1024^2\). A função max() mantém registrado o maior consumo observado durante a execução.
9. Avaliação dos modelos
Após o processamento, as classificações previstas são comparadas com os rótulos reais do conjunto de dados.
acuracia = accuracy_score(    rotulos_reais,    predicoes)precisao = precision_score(    rotulos_reais,    predicoes,    zero_division=0)recall = recall_score(    rotulos_reais,    predicoes,    zero_division=0)f1 = f1_score(    rotulos_reais,    predicoes,    zero_division=0)



São utilizadas quatro métricas para avaliar a qualidade das classificações: acurácia, precisão, recall e F1-score.

O parâmetro zero_division=0 estabelece que o resultado será zero caso ocorra uma situação em que a métrica não possa ser calculada devido à ausência de previsões positivas.

A taxa de processamento também é calculada considerando o total de inferências realizadas e o tempo registrado:
total_inferencias = len(textos) * PASSAGENStaxa_processamento = (    total_inferencias / tempo)



Dessa maneira, é possível comparar a velocidade de processamento dos modelos independentemente de suas diferenças de arquitetura.
10. Registro e consolidação dos resultados
Os indicadores obtidos são organizados em um dicionário:
resultado = {    "modelo": nome,    "identificador": identificador,    "repeticao": repeticao,    "amostras_unicas": len(textos),    "passagens": PASSAGENS,    "inferencias": total_inferencias,    "batch_size": BATCH_SIZE,    "max_length": MAX_LENGTH,    "tempo_segundos": tempo,    "inferencias_por_segundo": taxa_processamento,    "energia_kwh": energia_kwh,    "emissoes_kg_co2eq": emissoes,    "memoria_pico_mb": memoria_pico,    "acuracia": acuracia,    "precisao": precisao,    "recall": recall,    "f1": f1}



Cada execução é registrada no arquivo CSV individual, permitindo preservar os resultados sem sobrescrever as repetições anteriores.
pd.DataFrame([resultado]).to_csv(    ARQUIVO_RESULTADOS,    mode="a",    header=not os.path.exists(ARQUIVO_RESULTADOS),    index=False)



O parâmetro mode="a" permite adicionar novos registros ao arquivo, enquanto header controla a inclusão do cabeçalho somente quando necessário.
10.1. Resumo estatístico
Ao final de cada repetição, os resultados disponíveis são agrupados por modelo e submetidos ao cálculo de média e desvio padrão.
resumo = df_atual.groupby("modelo")[colunas_metricas].agg(    ["mean", "std"])



A média representa o comportamento médio observado nas repetições, enquanto o desvio padrão permite avaliar a variação dos resultados entre as execuções.

O resumo é exportado para resumo_modelos.csv, permitindo realizar comparações entre os modelos com base nos indicadores coletados.
11. Liberação de recursos e encerramento
Após concluir as cinco repetições de um modelo, os objetos utilizados são removidos:
del modelo, tokenizer, lotesgc.collect()



A remoção dos objetos e a execução do coletor de lixo auxiliam no gerenciamento da memória durante o processamento sequencial dos modelos.

Ao finalizar todos os modelos, o programa informa a conclusão do experimento e apresenta os caminhos dos arquivos contendo os resultados individuais e consolidados.

12. Estrutura dos arquivos gerados
Ao final da execução, o diretório de resultados apresenta uma organização semelhante à seguinte:
resultados_experimento_definitivo/
│
├── resultados_individuais.csv
├── resumo_modelos.csv
│
├── emissions_bert_tiny_rep_1.csv
├── emissions_bert_tiny_rep_2.csv
├── emissions_bert_tiny_rep_3.csv
├── emissions_bert_tiny_rep_4.csv
├── emissions_bert_tiny_rep_5.csv
│
├── emissions_bert_medium_rep_1.csv
├── emissions_distilbert_rep_1.csv
└── ...


Os arquivos de emissões são gerados individualmente pelo CodeCarbon, enquanto os outros dois arquivos são produzidos pelo próprio código experimental.
13. Considerações sobre os resultados
O experimento permite analisar a relação entre o desempenho dos modelos e os recursos computacionais necessários para sua execução.

A combinação dos indicadores de tempo, taxa de processamento, memória, energia e emissões estimadas possibilita identificar diferenças de eficiência entre as arquiteturas avaliadas.

As métricas de classificação complementam essa análise, permitindo verificar se as diferenças de desempenho computacional estão associadas a diferenças na qualidade das previsões.

Os resultados devem ser interpretados considerando as condições específicas do ambiente experimental, especialmente a utilização exclusiva de CPU, o conjunto de dados empregado e os parâmetros estabelecidos para o monitoramento ambiental.
