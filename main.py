import argparse

from config import RESULTS_DIR, LOGGER, criar_diretorios_projeto
from utils import RANDOM_STATE, DATASET_PATH
import data_processing
import evaluation


def run_pipeline(retrain_models, compare_train_test=False):
    criar_diretorios_projeto()
    LOGGER.info("--- INICIANDO PIPELINE DE PREDIÇÃO DE DIABETES ---")

    df = data_processing.carregar_dados(DATASET_PATH)
    if df is None:
        LOGGER.error("Falha ao carregar os dados. Abortando o pipeline.")
        return
    df = data_processing.analisar_dados(df)

    x_train, x_val, x_test, y_train, y_val, y_test, scaler, encoder, feature_names = data_processing.pre_processar_dados(df)

    tf_available = True
    modeling = None
    training = None
    load_model = None
    try:
        import tensorflow as _tf
        from tensorflow.keras.models import load_model as _load_model
        import modeling as _modeling
        import training as _training
        load_model = _load_model
        modeling = _modeling
        training = _training
    except Exception as e:
        tf_available = False
        LOGGER.warning(f"TensorFlow indisponível. Partes de deep learning serão ignoradas. Detalhe: {e}")

    if retrain_models and tf_available:
        LOGGER.info("--- FASE DE TREINAMENTO (Flag --retrain ativada) ---")
        classic_models = modeling.obter_modelos_classicos(RANDOM_STATE)
        training.treinar_modelos_classicos_pt(classic_models, x_train, y_train)
        modelo_mlp = modeling.criar_modelo_mlp_pt(input_shape=(x_train.shape[1],))
        training.treinar_modelo_keras_pt(modelo_mlp, x_train, y_train, x_val, y_val, "MLP")
        # modelo_cnn = modeling.criar_modelo_cnn_pt(input_shape=(x_train.shape[1], 1))
        # training.treinar_modelo_keras_pt(modelo_cnn, x_train, y_train, x_val, y_val, "CNN")
        # modelo_hibrido = modeling.criar_modelo_hibrido_pt(input_shape=(x_train.shape[1], 1))
        # training.treinar_modelo_keras_pt(modelo_hibrido, x_train, y_train, x_val, y_val, "Hibrido_CNN_LSTM")
    elif retrain_models and not tf_available:
        LOGGER.info("--- FASE DE TREINAMENTO (apenas modelos clássicos; TensorFlow ausente) ---")
        from modeling import obter_modelos_classicos
        from training import treinar_modelos_classicos_pt
        classic_models = obter_modelos_classicos(RANDOM_STATE)
        treinar_modelos_classicos_pt(classic_models, x_train, y_train)
    else:
        LOGGER.info("--- FASE DE TREINAMENTO PULADA (Usando modelos pré-treinados) ---")

    LOGGER.info("--- FASE DE AVALIAÇÃO ---")
    all_metrics = []

    from modeling import obter_modelos_classicos
    classic_for_names = obter_modelos_classicos(RANDOM_STATE)

    import os, pickle
    need_retrain_classics = False
    loaded_classic_models = {}

    for name in classic_for_names.keys():
        model_path = os.path.join(RESULTS_DIR, "modelos", f"{name.replace(' ', '_').lower()}.pkl")
        try:
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    loaded_classic_models[name] = pickle.load(f)
            else:
                LOGGER.warning(f"Modelo clássico {name} não encontrado em disco.")
                need_retrain_classics = True
        except Exception as e:
            LOGGER.warning(f"Falha ao carregar modelo clássico {name}: {e}")
            need_retrain_classics = True

    if need_retrain_classics:
        LOGGER.info("Retreinando modelos clássicos devido a ausência/erro de carga...")
        from training import treinar_modelos_classicos_pt
        classic_models = obter_modelos_classicos(RANDOM_STATE)
        treinar_modelos_classicos_pt(classic_models, x_train, y_train)
        loaded_classic_models = {}
        for name in classic_models.keys():
            model_path = os.path.join(RESULTS_DIR, "modelos", f"{name.replace(' ', '_').lower()}.pkl")
            with open(model_path, 'rb') as f:
                loaded_classic_models[name] = pickle.load(f)

    # Avaliação em TESTE (sempre)
    for name, model in loaded_classic_models.items():
        metrics = evaluation.avaliar_modelo(model, x_test, y_test, name, is_keras_model=False)
        all_metrics.append(metrics)

    if tf_available and load_model is not None:
        keras_models_to_evaluate = {
            "MLP": "MLP_best.keras",
            "CNN": "CNN_best.keras",
            "Hibrido_CNN_LSTM": "Hibrido_CNN_LSTM_best.keras"
        }
        import os
        for name, path in keras_models_to_evaluate.items():
            model_path = os.path.join(RESULTS_DIR, "modelos", path)
            if os.path.exists(model_path):
                best_model = load_model(model_path)
                metrics = evaluation.avaliar_modelo(best_model, x_test, y_test, name, is_keras_model=True)
                all_metrics.append(metrics)
            else:
                LOGGER.warning(f"Modelo Keras {name} não encontrado. Pule a avaliação ou execute com --retrain.")
    else:
        LOGGER.info("Avaliação de modelos Keras ignorada (TensorFlow indisponível).")

    if all_metrics:
        import pandas as pd
        metrics_df = pd.DataFrame(all_metrics)
        evaluation.comparar_todos_modelos(metrics_df)
    else:
        LOGGER.error("Nenhuma métrica foi gerada. Execute com a flag --retrain primeiro.")

    # Comparação Treino vs Teste (opcional via flag)
    if compare_train_test:
        import pandas as pd
        import gerar_graficos as gg
        train_metrics_rows = []
        test_metrics_rows = []

        # Clássicos: medir em train e test com nomes distintos para salvar gráficos sem sobrescrever
        for name, model in loaded_classic_models.items():
            try:
                m_train = evaluation.avaliar_modelo(model, x_train, y_train, f"{name}_train", is_keras_model=False)
                m_test = evaluation.avaliar_modelo(model, x_test, y_test, f"{name}_test", is_keras_model=False)
                # Ajustar 'modelo' base para alinhamento entre DataFrames
                m_train['modelo'] = name
                m_test['modelo'] = name
                train_metrics_rows.append(m_train)
                test_metrics_rows.append(m_test)
            except Exception as e:
                LOGGER.error(f"Falha ao comparar treino/teste para {name}: {e}")

        # Keras (se disponível): medir em train e test
        if tf_available and load_model is not None:
            keras_models_to_evaluate = {
                "MLP": "MLP_best.keras",
                "CNN": "CNN_best.keras",
                "Hibrido_CNN_LSTM": "Hibrido_CNN_LSTM_best.keras"
            }
            for name, path in keras_models_to_evaluate.items():
                model_path = os.path.join(RESULTS_DIR, "modelos", path)
                if not os.path.exists(model_path):
                    continue
                try:
                    best_model = load_model(model_path)
                    m_train = evaluation.avaliar_modelo(best_model, x_train, y_train, f"{name}_train", is_keras_model=True)
                    m_test = evaluation.avaliar_modelo(best_model, x_test, y_test, f"{name}_test", is_keras_model=True)
                    m_train['modelo'] = name
                    m_test['modelo'] = name
                    train_metrics_rows.append(m_train)
                    test_metrics_rows.append(m_test)
                except Exception as e:
                    LOGGER.error(f"Falha ao comparar treino/teste para {name} (Keras): {e}")
        else:
            LOGGER.info("Comparação treino/teste para modelos Keras ignorada (TensorFlow indisponível).")

        if train_metrics_rows and test_metrics_rows:
            df_train = pd.DataFrame(train_metrics_rows)
            df_test = pd.DataFrame(test_metrics_rows)
            gg.visualizar_comparacao_treino_teste(df_train, df_test)
            LOGGER.info("Comparação Treino vs Teste gerada com sucesso.")
        else:
            LOGGER.warning("Não foi possível gerar comparação Treino vs Teste (sem métricas).")

    LOGGER.info("--- PIPELINE CONCLUÍDO ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de Treinamento e Avaliação para Predição de Diabetes.")
    parser.add_argument('--retrain', action='store_true', help="Força retreinamento de todos os modelos.")
    # parser.add_argument('--comparacao-treino-teste', action='store_true', help="Gera gráficos comparando métricas no treino vs teste.")
    args = parser.parse_args()
    run_pipeline(retrain_models=args.retrain, compare_train_test=args.retrain)
