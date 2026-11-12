import os
import gc
import time
import torch
import psutil
import pandas as pd

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from codecarbon import EmissionsTracker
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

MODELOS = {
    "BERT Tiny": "takedarn/bert-tiny-sst2",
    "BERT Medium": "takedarn/bert-medium-sst2",
    "DistilBERT": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    "BERT Base": "yoshitomo-matsubara/bert-base-uncased-sst2"
}

BATCH_SIZE = 8
MAX_LENGTH = 128
PASSAGENS = 10
REPETICOES = 5

FATOR_CARBONO = 98.348

PASTA_RESULTADOS = "resultados_experimento_definitivo"

ARQUIVO_RESULTADOS = os.path.join(
    PASTA_RESULTADOS, "resultados_individuais.csv"
)

ARQUIVO_RESUMO = os.path.join(
    PASTA_RESULTADOS, "resumo_modelos.csv"
)

os.makedirs(PASTA_RESULTADOS, exist_ok=True)

if os.path.exists(ARQUIVO_RESULTADOS):
    os.remove(ARQUIVO_RESULTADOS)

torch.set_num_threads(4)
torch.set_num_interop_threads(4)

processo = psutil.Process(os.getpid())

dataset = load_dataset("stanfordnlp/sst2")["validation"]

textos = dataset["sentence"]
rotulos_reais = dataset["label"]

resultados = []

for nome, identificador in MODELOS.items():

    print(f"\n{'=' * 50}")
    print(f"Modelo: {nome}")
    print(f"Identificador: {identificador}")
    print(f"{'=' * 50}")

    tokenizer = AutoTokenizer.from_pretrained(identificador)

    modelo = AutoModelForSequenceClassification.from_pretrained(
        identificador
    )

    modelo.to("cpu")
    modelo.eval()

    lotes = []

    for inicio in range(0, len(textos), BATCH_SIZE):

        entradas = tokenizer(
            textos[inicio:inicio + BATCH_SIZE],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )

        lotes.append(entradas)

    for repeticao in range(1, REPETICOES + 1):

        print(f"\nRepetição {repeticao}/{REPETICOES}")

        with torch.inference_mode():
            for lote in lotes:
                modelo(**lote)

        predicoes = []

        memoria_pico = (
            processo.memory_info().rss / (1024 ** 2)
        )

        nome_arquivo = (
            f"emissions_{nome.lower().replace(' ', '_')}_"
            f"rep_{repeticao}.csv"
        )

        tracker = EmissionsTracker(
            project_name=f"TCC_{nome.replace(' ', '_')}_R{repeticao}",
            measure_power_secs=1,
            tracking_mode="process",
            pue=1.0,
            wue=0,
            api_call_interval=-1,
            force_carbon_intensity_g_co2e_kwh=FATOR_CARBONO,
            output_dir=PASTA_RESULTADOS,
            output_file=nome_arquivo
        )

        tracker.start()

        inicio = time.perf_counter()

        with torch.inference_mode():

            for passagem in range(PASSAGENS):

                for lote in lotes:

                    saida = modelo(**lote)

                    if passagem == 0:

                        predicoes.extend(
                            torch.argmax(
                                saida.logits,
                                dim=1
                            ).cpu().tolist()
                        )

                    memoria_atual = (
                        processo.memory_info().rss / (1024 ** 2)
                    )

                    memoria_pico = max(
                        memoria_pico,
                        memoria_atual
                    )

        tempo = time.perf_counter() - inicio

        emissoes = tracker.stop()

        energia_kwh = float("nan")

        try:

            energia_kwh = float(
                tracker.final_emissions_data.energy_consumed
            )

        except Exception:

            try:

                dados_codecarbon = pd.read_csv(
                    os.path.join(
                        PASTA_RESULTADOS,
                        nome_arquivo
                    )
                )

                energia_kwh = float(
                    dados_codecarbon["energy_consumed"].iloc[-1]
                )

            except Exception:
                pass

        total_inferencias = len(textos) * PASSAGENS

        taxa_processamento = (
            total_inferencias / tempo
        )

        acuracia = accuracy_score(
            rotulos_reais,
            predicoes
        )

        precisao = precision_score(
            rotulos_reais,
            predicoes,
            zero_division=0
        )

        recall = recall_score(
            rotulos_reais,
            predicoes,
            zero_division=0
        )

        f1 = f1_score(
            rotulos_reais,
            predicoes,
            zero_division=0
        )

        resultado = {
            "modelo": nome,
            "identificador": identificador,
            "repeticao": repeticao,
            "amostras_unicas": len(textos),
            "passagens": PASSAGENS,
            "inferencias": total_inferencias,
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "tempo_segundos": tempo,
            "inferencias_por_segundo": taxa_processamento,
            "energia_kwh": energia_kwh,
            "emissoes_kg_co2eq": emissoes,
            "memoria_pico_mb": memoria_pico,
            "acuracia": acuracia,
            "precisao": precisao,
            "recall": recall,
            "f1": f1
        }

        resultados.append(resultado)

        pd.DataFrame([resultado]).to_csv(
            ARQUIVO_RESULTADOS,
            mode="a",
            header=not os.path.exists(ARQUIVO_RESULTADOS),
            index=False
        )

        print(f"Tempo: {tempo:.2f} segundos")
        print(f"Taxa: {taxa_processamento:.2f} inferências/s")
        print(f"Energia estimada: {energia_kwh} kWh")
        print(f"Emissões estimadas: {emissoes} kg CO2eq")
        print(f"Memória máxima observada: {memoria_pico:.2f} MB")
        print(f"Acurácia: {acuracia:.4f}")
        print(f"Precisão: {precisao:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")

        df_atual = pd.DataFrame(resultados)

        colunas_metricas = [
            "tempo_segundos",
            "inferencias_por_segundo",
            "energia_kwh",
            "emissoes_kg_co2eq",
            "memoria_pico_mb",
            "acuracia",
            "precisao",
            "recall",
            "f1"
        ]

        resumo = df_atual.groupby("modelo")[colunas_metricas].agg(
            ["mean", "std"]
        )

        resumo.columns = [
            f"{coluna}_{estatistica}"
            for coluna, estatistica in resumo.columns
        ]

        resumo.to_csv(ARQUIVO_RESUMO)

    del modelo, tokenizer, lotes
    gc.collect()

print("\nExperimento concluído.")
print(f"Resultados individuais: {ARQUIVO_RESULTADOS}")
print(f"Resumo estatístico: {ARQUIVO_RESUMO}")
