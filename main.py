import pickle

import pandas as pd
from tensorflow.keras.models import load_model
import os
import numpy as np
import argparse

from config import RESULTS_DIR, LOGGER, criar_diretorios_projeto
from utils import RANDOM_STATE, DATASET_PATH

import data_processing
import modeling
import training
import evaluation
import interpretability


def run_pipeline(retrain_models):
    """
    Executa o pipeline de Machine Learning.

    Args:
        retrain_models (bool): Se True, treina os modelos do zero.
                               Se False, pula o treino e usa os modelos já salvos.
    """

    criar_diretorios_projeto()
    LOGGER.info("--- INICIANDO PIPELINE DE PREDIÇÃO DE DIABETES ---")

    # 1. Carga e Análise dos Dados
    df = data_processing.carregar_dados(DATASET_PATH)
    if df is None:
        LOGGER.error("Falha ao carregar os dados. Abortando o pipeline.")
        return
    df = data_processing.analisar_dados(df)

    # 2. Pré-processamento
    (x_train, x_val, x_test, y_train, y_val, y_test,
     scaler, encoder, feature_names) = data_processing.pre_processar_dados(df)

    # 3. Fase de Treinamento (Condicional)
    if retrain_models:
        LOGGER.info("--- FASE DE TREINAMENTO (Flag --retrain ativada) ---")

        # a) Modelos Clássicos
        classic_models = modeling.obter_modelos_classicos(RANDOM_STATE)
        training.treinar_modelos_classicos_pt(classic_models, x_train, y_train)

        # b) Modelo MLP
        modelo_mlp = modeling.criar_modelo_mlp_pt(input_shape=(x_train.shape[1],))
        training.treinar_modelo_keras_pt(modelo_mlp, x_train, y_train, x_val, y_val, "MLP")

        # c) Modelo CNN
        modelo_cnn = modeling.criar_modelo_cnn_pt(input_shape=(x_train.shape[1], 1))
        training.treinar_modelo_keras_pt(modelo_cnn, x_train, y_train, x_val, y_val, "CNN")

        # d) Modelo Híbrido
        modelo_hibrido = modeling.criar_modelo_hibrido_pt(input_shape=(x_train.shape[1], 1))
        training.treinar_modelo_keras_pt(modelo_hibrido, x_train, y_train, x_val, y_val, "Hibrido_CNN_LSTM")
    else:
        LOGGER.info("--- FASE DE TREINAMENTO PULADA (Usando modelos pré-treinados) ---")

    # 4. Avaliação (Sempre executa, usando os modelos salvos)
    LOGGER.info("--- FASE DE AVALIAÇÃO ---")
    all_metrics = []

    # a) Avaliar modelos clássicos
    trained_classic_models = modeling.obter_modelos_classicos(RANDOM_STATE)  # Apenas para ter a lista de nomes
    for name in trained_classic_models.keys():
        model_path = os.path.join(RESULTS_DIR, "modelos", f"{name.replace(' ', '_').lower()}.pkl")
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            metrics = evaluation.avaliar_modelo(model, x_test, y_test, name, is_keras_model=False)
            all_metrics.append(metrics)
        else:
            LOGGER.warning(f"Modelo clássico {name} não encontrado. Pule a avaliação ou execute com --retrain.")

    # b) Avaliar modelos Keras
    keras_models_to_evaluate = {
        "MLP": "MLP_best.keras",
        "CNN": "CNN_best.keras",
        "Hibrido_CNN_LSTM": "Hibrido_CNN_LSTM_best.keras"
    }

    for name, path in keras_models_to_evaluate.items():
        model_path = os.path.join(RESULTS_DIR, "modelos", path)
        if os.path.exists(model_path):
            best_model = load_model(model_path)
            metrics = evaluation.avaliar_modelo(best_model, x_test, y_test, name, is_keras_model=True)
            all_metrics.append(metrics)
        else:
            LOGGER.warning(f"Modelo Keras {name} não encontrado. Pule a avaliação ou execute com --retrain.")

    # c) Comparar todos os modelos
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        evaluation.comparar_todos_modelos(metrics_df)
    else:
        LOGGER.error("Nenhuma métrica foi gerada. Impossível comparar modelos. Execute com a flag --retrain primeiro.")

    # 5. Interpretabilidade
    LOGGER.info("--- FASE DE INTERPRETABILIDADE ---")

    # Dicionário para iterar sobre os modelos que queremos interpretar
    modelos_para_interpretar = {
        "MLP": "MLP_best.keras",
        "CNN": "CNN_best.keras",
        "Hibrido_CNN_LSTM": "Hibrido_CNN_LSTM_best.keras"
    }

    for nome_modelo, caminho_modelo in modelos_para_interpretar.items():
        LOGGER.info(f"==> Iniciando análise de interpretabilidade para o modelo: {nome_modelo} <==")

        model_path = os.path.join(RESULTS_DIR, "modelos", caminho_modelo)

        if os.path.exists(model_path):
            best_model = load_model(model_path)

            # Prepara os dados de teste (reshape para CNN/Híbrido se necessário)
            x_test_para_analise = x_test
            if "CNN" in nome_modelo or "Hibrido" in nome_modelo:
                if len(x_test_para_analise.shape) == 2:
                    x_test_para_analise = np.expand_dims(x_test_para_analise, axis=-1)

            # Executa as análises de interpretabilidade
            interpretability.analisar_valores_shap(best_model, x_test_para_analise, feature_names, nome_modelo)
            interpretability.analisar_lime_pt(best_model, x_train, x_test_para_analise, feature_names, nome_modelo)
        else:
            LOGGER.warning(f"Modelo {nome_modelo} não encontrado em {model_path}. Pulando análise.")

    LOGGER.info("--- PIPELINE CONCLUÍDO ---")


if __name__ == "__main__":
    # Configuração do parser de argumentos
    parser = argparse.ArgumentParser(description="Pipeline de Treinamento e Avaliação para Predição de Diabetes.")
    parser.add_argument(
        '--retrain',
        action='store_true',
        help="Se especificado, força o retreinamento de todos os modelos. Caso contrário, usa os modelos já salvos."
    )
    args = parser.parse_args()

    # Executa o pipeline com a flag
    run_pipeline(retrain_models=args.retrain)
