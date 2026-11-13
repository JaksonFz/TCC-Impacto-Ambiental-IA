

## Estrutura e inicialização do programa

O script foi desenvolvido em Python com a finalidade de executar um experimento comparativo entre quatro modelos Transformer utilizados na classificação de sentimentos. Sua estrutura foi organizada para automatizar o carregamento dos modelos, a preparação dos dados, a execução repetida das inferências, o monitoramento dos recursos computacionais e o armazenamento dos resultados obtidos.

Inicialmente, são importadas as bibliotecas necessárias para a execução dessas operações. A biblioteca `os` é utilizada para manipular diretórios e caminhos de arquivos, permitindo organizar automaticamente os resultados produzidos pelo experimento. A biblioteca `gc` auxilia no gerenciamento da memória, possibilitando solicitar a liberação de objetos que não são mais necessários após a execução de cada modelo. Já a biblioteca `time` fornece a função `perf_counter()`, utilizada posteriormente para medir o tempo de processamento com maior precisão.

O PyTorch, importado pelo módulo `torch`, constitui a base de execução dos modelos, sendo responsável pelo processamento dos tensores e pela realização das operações de inferência. O módulo `psutil` permite acessar informações sobre o processo Python em execução, especialmente o consumo de memória RAM. Para organizar os dados e exportar os resultados, utiliza-se a biblioteca Pandas. As bibliotecas `datasets` e `transformers`, por sua vez, são responsáveis pelo acesso ao conjunto de dados SST-2 e pelo carregamento dos modelos pré-treinados e seus respectivos tokenizadores.

O monitoramento ambiental é realizado por meio da biblioteca CodeCarbon, enquanto as métricas de classificação são calculadas com funções disponibilizadas pelo Scikit-learn.

Após as importações, o código define um dicionário denominado `MODELOS`. Essa estrutura associa um nome simplificado a cada modelo e ao respectivo identificador utilizado na plataforma Hugging Face. Essa abordagem permite percorrer todos os modelos por meio de uma única estrutura de repetição, evitando a necessidade de implementar separadamente o mesmo procedimento para cada arquitetura.

Em seguida, são definidas as constantes que controlam o experimento. A variável `BATCH_SIZE` estabelece que as sentenças serão processadas em grupos de oito amostras, enquanto `MAX_LENGTH` determina o limite de 128 tokens para cada sequência de entrada. A variável `PASSAGENS` define dez percursos completos pelo conjunto de dados em cada repetição, e `REPETICOES` estabelece cinco execuções independentes para cada modelo.

Também é definido o parâmetro `FATOR_CARBONO`, correspondente à intensidade de carbono utilizada pelo CodeCarbon, expressa em gramas de CO₂ equivalente por quilowatt-hora. Os caminhos dos arquivos de saída são construídos com `os.path.join()`, centralizando os resultados no diretório `resultados_experimento_definitivo`.

Antes do início do processamento, `os.makedirs()` cria esse diretório caso ele ainda não exista. O código também verifica a existência do arquivo de resultados individuais e o remove quando encontrado. Essa operação evita que registros de execuções anteriores sejam incorporados aos resultados da execução atual.

Por fim, são configuradas quatro threads para as operações intraoperação e interoperação do PyTorch. Essa configuração estabelece a quantidade de threads utilizada pelo framework nessas categorias de operações, buscando padronizar as condições computacionais entre os modelos. A função `psutil.Process(os.getpid())` também é utilizada para obter uma referência ao processo Python atual, que será consultada durante o monitoramento da memória.

## Carregamento e preparação do conjunto de dados

Após a inicialização, o programa carrega o conjunto de dados SST-2 por meio da função `load_dataset()`, disponibilizada pela biblioteca Datasets. O código utiliza especificamente a divisão de validação, identificada pelo parâmetro `validation`.

A partir do conjunto carregado, são extraídas duas listas: `textos` e `rotulos_reais`. A primeira contém as sentenças em inglês que serão submetidas aos modelos, enquanto a segunda armazena os rótulos correspondentes às classificações de referência. No SST-2, essas classificações representam sentimentos positivos e negativos.

A separação entre textos e rótulos permite utilizar os mesmos dados de entrada para todos os modelos, garantindo que as comparações sejam realizadas sobre o mesmo conjunto de amostras. Os rótulos reais permanecem armazenados para serem utilizados posteriormente na avaliação das previsões.

## Carregamento dos modelos e tokenização

A execução principal é organizada por meio de uma estrutura de repetição que percorre os elementos do dicionário `MODELOS`. Em cada iteração, o programa obtém o nome e o identificador do modelo que será avaliado.

Primeiramente, são exibidas no terminal informações de identificação, permitindo acompanhar qual modelo está sendo processado. Em seguida, a função `AutoTokenizer.from_pretrained()` carrega o tokenizador correspondente ao modelo selecionado. O tokenizador é responsável por converter as sentenças textuais em representações numéricas que possam ser interpretadas pela arquitetura Transformer.

Na sequência, `AutoModelForSequenceClassification.from_pretrained()` carrega o modelo pré-treinado para classificação de sequências. Essa classe permite utilizar diferentes arquiteturas por meio de uma interface comum, desde que sejam compatíveis com a tarefa de classificação.

Após o carregamento, a instrução `modelo.to("cpu")` direciona os parâmetros do modelo para a unidade central de processamento. Essa decisão mantém a execução em ambiente exclusivamente baseado em CPU, sem utilização de GPU. A chamada `modelo.eval()` coloca a rede neural em modo de avaliação, desativando comportamentos específicos do treinamento, como o dropout aplicado durante essa etapa.

Antes de iniciar as repetições, o código prepara os lotes de dados. Para isso, percorre a lista de sentenças utilizando um intervalo definido por `BATCH_SIZE`. Em cada iteração, o tokenizador recebe um subconjunto de textos e realiza sua conversão por meio de diferentes parâmetros.

O argumento `return_tensors="pt"` determina que as representações resultantes sejam retornadas como tensores compatíveis com o PyTorch. O parâmetro `padding="max_length"` realiza o preenchimento das sequências até o comprimento máximo estabelecido, enquanto `truncation=True` permite remover tokens excedentes. Já `max_length=MAX_LENGTH` limita cada sequência a 128 tokens.

Os lotes resultantes são armazenados na lista `lotes`. Dessa forma, a tokenização é realizada apenas uma vez por modelo, e os mesmos tensores preparados são reutilizados nas cinco repetições. Essa decisão evita incluir repetidamente o custo de tokenização na medição principal das inferências.

## Execução das repetições e aquecimento

Com o modelo carregado e os lotes preparados, inicia-se a estrutura de repetição responsável por executar cinco avaliações independentes. A variável `repeticao` identifica cada execução, variando de 1 a 5.

Antes de cada execução monitorada, o código realiza uma passagem completa pelos lotes dentro de um bloco `torch.inference_mode()`. Essa etapa corresponde ao aquecimento (*warm-up*) do modelo. Sua finalidade é permitir que as operações iniciais sejam realizadas antes do início da coleta das métricas principais.

O uso de `torch.inference_mode()` desativa o acompanhamento de gradientes e outras operações relacionadas à diferenciação automática, uma vez que o experimento não envolve treinamento ou atualização dos parâmetros da rede neural. Isso reduz o trabalho adicional do framework durante a inferência.

Após o aquecimento, a lista `predicoes` é inicializada para armazenar as classificações produzidas pelo modelo. O consumo inicial de memória do processo também é registrado por meio de `processo.memory_info().rss`. O atributo RSS (*Resident Set Size*) representa a quantidade de memória física residente associada ao processo Python naquele instante. O valor é convertido de bytes para megabytes por meio da divisão por `1024 ** 2`.

O nome do arquivo de monitoramento ambiental é construído dinamicamente com base no modelo e no número da repetição. Essa identificação permite associar cada arquivo de emissões à execução correspondente, evitando a sobreposição dos registros.

## Monitoramento ambiental e medição do tempo

Para monitorar o consumo energético estimado, o código instancia a classe `EmissionsTracker`, disponibilizada pelo CodeCarbon. Cada instância recebe um nome de projeto que identifica o modelo e sua respectiva repetição, além das configurações de monitoramento.

O parâmetro `measure_power_secs=1` estabelece o intervalo de amostragem utilizado pelo monitoramento de potência. A configuração `tracking_mode="process"` determina que o acompanhamento seja direcionado ao processo em execução. O parâmetro `pue=1.0` define o valor de eficiência de uso de energia da infraestrutura, enquanto `wue=0` desativa a contabilização de água nessa configuração.

A opção `force_carbon_intensity_g_co2e_kwh=FATOR_CARBONO` estabelece uma intensidade de carbono fixa para o cálculo das emissões. Com isso, evita-se que diferentes execuções utilizem fatores de intensidade de carbono distintos durante a comparação.

Também são definidos o diretório de saída e o nome do arquivo CSV que receberá os dados do monitoramento. O parâmetro `api_call_interval=-1` desativa as consultas periódicas à API de intensidade de carbono, sendo utilizado em conjunto com o fator fixo informado.

Após a configuração, `tracker.start()` inicia o monitoramento ambiental. Imediatamente depois, `time.perf_counter()` registra o instante inicial da medição de tempo.

A execução das inferências ocorre dentro de um segundo bloco `torch.inference_mode()`. Nesse momento, uma estrutura de repetição percorre dez vezes a lista completa de lotes. Em cada passagem, o modelo recebe um lote por meio da chamada `modelo(**lote)`, que encaminha os tensores de entrada aos argumentos esperados pela arquitetura.

O resultado retornado contém, entre outras informações, os valores de saída da rede neural, armazenados no atributo `logits`. Esses valores representam as pontuações produzidas pelo modelo para cada classe.

Durante a primeira passagem, o código utiliza `torch.argmax(saida.logits, dim=1)` para identificar a classe com maior pontuação em cada amostra. A operação `argmax` retorna o índice da maior pontuação ao longo da dimensão das classes, produzindo assim a classificação prevista pelo modelo.

As previsões são transferidas para a memória principal por meio de `.cpu()` e convertidas em uma lista Python com `.tolist()`. O uso da condição `if passagem == 0` garante que as previsões sejam armazenadas apenas na primeira passagem, evitando duplicações na avaliação da classificação.

Nas dez passagens, o código também consulta continuamente o consumo de memória do processo. O valor atual de RSS é comparado com o maior valor registrado anteriormente por meio da função `max()`. Dessa maneira, `memoria_pico` mantém o maior consumo de memória observado durante a execução monitorada, incluindo o valor inicial registrado antes do início das inferências.

Quando todas as passagens são concluídas, `time.perf_counter()` é chamado novamente. A diferença entre o instante final e o inicial representa o tempo total de processamento das inferências. Em seguida, `tracker.stop()` encerra o monitoramento do CodeCarbon e retorna a estimativa de emissões de CO₂ equivalente da execução.

## Recuperação dos dados energéticos

Após o encerramento do monitoramento, o código inicializa a variável `energia_kwh` com um valor inválido (`NaN`). Essa inicialização permite identificar situações em que não seja possível recuperar a informação energética.

Primeiramente, o programa tenta acessar diretamente o atributo `energy_consumed`, presente em `tracker.final_emissions_data`. Caso essa operação seja bem-sucedida, o valor é convertido para o tipo numérico `float`.

Se ocorrer alguma exceção durante essa tentativa, o código utiliza uma alternativa: abre o arquivo CSV produzido pelo CodeCarbon por meio do Pandas e procura a coluna `energy_consumed`. O último registro dessa coluna é então utilizado como valor de consumo energético.

Essa estrutura de tratamento de exceções oferece uma alternativa para recuperar a informação caso o acesso direto ao objeto do monitoramento não esteja disponível. Caso ambas as tentativas falhem, o valor permanece como `NaN`, indicando que o consumo energético não pôde ser recuperado.

## Cálculo das métricas de desempenho e classificação

Com o encerramento das inferências, o programa calcula a quantidade total de operações realizadas durante a repetição. Essa quantidade é obtida multiplicando o número de sentenças do conjunto de dados pelo número de passagens:

`total_inferencias = len(textos) * PASSAGENS`

Como são utilizadas dez passagens completas, cada amostra é submetida ao modelo dez vezes. A taxa de processamento é calculada dividindo esse total pelo tempo medido:

`taxa_processamento = total_inferencias / tempo`

O resultado representa a quantidade de inferências realizadas por segundo durante o intervalo monitorado.

Na sequência, são calculadas as métricas de classificação. A função `accuracy_score()` compara as previsões armazenadas com os rótulos reais e determina a proporção de classificações corretas. A função `precision_score()` calcula a precisão, enquanto `recall_score()` determina a revocação. Por fim, `f1_score()` calcula a média harmônica entre precisão e revocação.

Nas quatro funções é utilizado o argumento `zero_division=0`, que define o valor retornado quando ocorre uma situação de divisão por zero durante o cálculo de uma métrica.

É importante observar que as métricas de classificação são calculadas sobre as previsões da primeira passagem, correspondentes às amostras únicas do conjunto de validação. Já o tempo e a taxa de processamento consideram as dez passagens completas. Essa separação permite avaliar a qualidade das classificações sem contabilizar repetidamente as mesmas amostras.

## Organização e persistência dos resultados

Após os cálculos, o código reúne todas as informações da repetição em um dicionário denominado `resultado`. Esse objeto concentra a identificação do modelo, os parâmetros utilizados, as métricas computacionais, as estimativas ambientais e os indicadores de classificação.

O dicionário é acrescentado à lista `resultados`, que mantém em memória os registros produzidos durante a execução dos modelos.

Em seguida, o Pandas transforma o dicionário em um DataFrame e o exporta para o arquivo `resultados_individuais.csv`. A gravação utiliza o modo de anexação (`mode="a"`), permitindo acrescentar novas linhas ao arquivo sem apagar os registros anteriores. O parâmetro `header` verifica a existência do arquivo para determinar se os nomes das colunas devem ser escritos novamente. Assim, o cabeçalho é inserido somente na primeira gravação.

Depois da persistência dos resultados individuais, o código cria outro DataFrame a partir da lista acumulada `resultados`. São selecionadas as colunas que representam as métricas de interesse, excluindo os campos de identificação e configuração que não participam do resumo estatístico.

O agrupamento é realizado por meio de `groupby("modelo")`, reunindo as repetições correspondentes a cada modelo. Sobre cada grupo, a função `agg(["mean", "std"])` calcula a média e o desvio-padrão das métricas selecionadas.

Como o agrupamento produz colunas com múltiplos níveis, o código utiliza uma compreensão de listas para combinar o nome de cada métrica com a estatística correspondente. Por exemplo, as colunas resultantes recebem identificações como `tempo_segundos_mean` e `tempo_segundos_std`.

Por fim, o resumo é exportado para `resumo_modelos.csv`. Esse arquivo permite consultar a média e a dispersão dos resultados obtidos nas cinco repetições de cada modelo, servindo como base para as análises comparativas e a elaboração de tabelas e gráficos.

## Finalização e liberação de recursos

Após concluir as cinco repetições de um modelo, o código utiliza `del` para remover as referências ao modelo, ao tokenizador e à lista de lotes. Em seguida, `gc.collect()` solicita a execução do coletor de lixo do Python, buscando liberar objetos que não possuem mais referências e recuperar memória antes do carregamento do próximo modelo.

Esse procedimento é importante porque os modelos Transformer podem ocupar quantidades consideráveis de memória, especialmente quando diferentes arquiteturas são executadas sequencialmente no mesmo processo.

Quando todos os modelos são processados, o programa exibe a mensagem de conclusão e informa os caminhos dos arquivos CSV gerados. Dessa forma, o experimento termina com os registros individuais de cada repetição e o resumo estatístico consolidado por modelo.

## Limitações e interpretação dos dados

Os resultados produzidos pelo script devem ser interpretados considerando as características do método de coleta. O CodeCarbon fornece estimativas de consumo energético e emissões com base no monitoramento do processo e nas informações de hardware e intensidade de carbono disponíveis. Portanto, esses valores não representam medições físicas diretas realizadas por um medidor externo.

Da mesma forma, o consumo de memória registrado corresponde ao RSS do processo Python, incluindo estruturas de dados, bibliotecas e demais componentes residentes na memória, não sendo uma medição isolada do espaço ocupado pelos parâmetros do modelo.

A medição principal contempla o processamento das dez passagens de inferência, mas não inclui o tempo de carregamento dos modelos, a tokenização e a preparação dos lotes. O aquecimento também é realizado antes do início da medição. Consequentemente, os resultados representam o custo computacional da etapa de inferência nas condições estabelecidas, e não o custo total de inicialização e utilização do sistema.

Por fim, a configuração `wue=0` desativa a contabilização de água pelo CodeCarbon nessa execução. Assim, os resultados desse script não devem ser interpretados como uma medição direta ou uma estimativa completa do consumo hídrico associado aos modelos.
