import os
import pickle
import glob
import pandas as pd
from config import RESULTS_DIR, LOGGER
from data_processing import carregar_dados, pre_processar_dados
from evaluation import avaliar_modelo
import gerar_graficos as gg


def load_models_from_dir(models_dir):
    models = {}

    for p in glob.glob(os.path.join(models_dir, "*.pkl")):
        name = os.path.splitext(os.path.basename(p))[0]
        try:
            with open(p, 'rb') as f:
                models[name] = pickle.load(f)
        except Exception as e:
            LOGGER.error(f"Erro ao carregar modelo pickle {p}: {e}")

    try:
        from tensorflow.keras.models import load_model
        for p in glob.glob(os.path.join(models_dir, "*.keras")):
            name = os.path.splitext(os.path.basename(p))[0]
            try:
                models[name] = load_model(p)
            except Exception as e:
                LOGGER.error(f"Erro ao carregar modelo keras {p}: {e}")
    except Exception as e:
        LOGGER.warning(f"TensorFlow não disponível para carregar modelos Keras: {e}")
    return models


if __name__ == '__main__':
    LOGGER.info("Iniciando geração de comparação Treino x Teste automática")
    data_path = os.path.join(os.getcwd(), '../diabetes_prediction_dataset.csv')
    df = carregar_dados(data_path)
    if df is None:
        LOGGER.error('Dataset não encontrado ou falha ao carregar. Abortando.')
        raise SystemExit(1)

    # Pré-processar dados
    X_train_res, X_val, X_test, y_train_res, y_val, y_test, scaler, encoder, feature_names = pre_processar_dados(df)

    # Carregar modelos salvos
    models_dir = os.path.join("../"+ RESULTS_DIR, 'modelos')
    models = load_models_from_dir(models_dir)
    if not models:
        LOGGER.error('Nenhum modelo encontrado em RESULTS_DIR/modelos')
        raise SystemExit(1)

    metrics_train = []
    metrics_test = []

    for name, model in models.items():
        LOGGER.info(f'Avaliando modelo: {name} (treino e teste)')
        is_keras = False
        try:
            from tensorflow.keras.models import Model
            is_keras = isinstance(model, Model)
        except Exception:
            is_keras = False

        try:
            m_train = avaliar_modelo(model, X_train_res, y_train_res, f"{name}_train", is_keras_model=is_keras)
            m_test = avaliar_modelo(model, X_test, y_test, f"{name}_test", is_keras_model=is_keras)
        except Exception as e:
            LOGGER.error(f'Erro avaliando modelo {name}: {e}')
            continue

        # Normalizar nome do modelo para coluna 'modelo'
        m_train['modelo'] = name
        m_test['modelo'] = name
        metrics_train.append(m_train)
        metrics_test.append(m_test)

    if not metrics_train or not metrics_test:
        LOGGER.error('Nenhuma métrica coletada. Verifique logs.')
        raise SystemExit(1)

    df_train = pd.DataFrame(metrics_train)
    df_test = pd.DataFrame(metrics_test)

    # Gerar gráficos comparativos
    gg.visualizar_comparacao_treino_teste(df_train, df_test)
    LOGGER.info('Gráficos Treino vs Teste gerados com sucesso')

