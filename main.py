import argparse
import os
import pickle
import pandas as pd

from config import RESULTS_DIR, LOGGER, criar_diretorios_projeto
from utils import RANDOM_STATE, DATASET_PATH
import data_processing
import evaluation


def check_tensorflow_availability():
    try:
        import tensorflow
        from tensorflow.keras.models import load_model
        import modeling
        import training
        return True, load_model, modeling, training
    except Exception as e:
        LOGGER.warning(f"TensorFlow indisponível. Partes de deep learning serão ignoradas. Detalhe: {e}")
        return False, None, None, None


def train_classic_models(x_train, y_train):
    from modeling import obter_modelos_classicos
    from training import treinar_modelos_classicos_pt

    LOGGER.info("Treinando modelos clássicos...")
    classic_models = obter_modelos_classicos(RANDOM_STATE)
    treinar_modelos_classicos_pt(classic_models, x_train, y_train)


def train_deep_learning_models(modeling, training, x_train, y_train, x_val, y_val):
    from gerar_graficos import visualizar_historico_treinamento

    LOGGER.info("Treinando modelos de deep learning...")

    modelo_mlp = modeling.criar_modelo_mlp_pt(input_shape=(x_train.shape[1],))
    modelo_mlp, hist_mlp = training.treinar_modelo_keras_pt(modelo_mlp, x_train, y_train, x_val, y_val, "MLP")
    visualizar_historico_treinamento(hist_mlp, "MLP")

    modelo_cnn = modeling.criar_modelo_cnn_pt(input_shape=(x_train.shape[1], 1))
    modelo_cnn, hist_cnn = training.treinar_modelo_keras_pt(modelo_cnn, x_train, y_train, x_val, y_val, "CNN")
    visualizar_historico_treinamento(hist_cnn, "CNN")


def load_classic_models(x_train, y_train):
    from modeling import obter_modelos_classicos
    from training import treinar_modelos_classicos_pt

    classic_for_names = obter_modelos_classicos(RANDOM_STATE)
    loaded_models = {}
    need_retrain = False

    for name in classic_for_names.keys():
        model_path = os.path.join(RESULTS_DIR, "modelos", f"{name.replace(' ', '_').lower()}.pkl")
        try:
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    loaded_models[name] = pickle.load(f)
            else:
                LOGGER.warning(f"Modelo clássico {name} não encontrado em disco.")
                need_retrain = True
        except Exception as e:
            LOGGER.warning(f"Falha ao carregar modelo clássico {name}: {e}")
            need_retrain = True

    if need_retrain:
        LOGGER.info("Retreinando modelos clássicos devido a ausência/erro de carga...")
        train_classic_models(x_train, y_train)
        loaded_models = {}
        for name in classic_for_names.keys():
            model_path = os.path.join(RESULTS_DIR, "modelos", f"{name.replace(' ', '_').lower()}.pkl")
            with open(model_path, 'rb') as f:
                loaded_models[name] = pickle.load(f)

    return loaded_models


def evaluate_classic_models(loaded_models, x_test, y_test):
    all_metrics = []
    for name, model in loaded_models.items():
        metrics = evaluation.avaliar_modelo(model, x_test, y_test, name, is_keras_model=False)
        all_metrics.append(metrics)
    return all_metrics


def evaluate_keras_models(load_model, x_test, y_test):
    keras_models = {
        "MLP": "MLP_best.keras",
        "CNN": "CNN_best.keras",
        "Hibrido_CNN_LSTM": "Hibrido_CNN_LSTM_best.keras"
    }

    all_metrics = []
    for name, filename in keras_models.items():
        model_path = os.path.join(RESULTS_DIR, "modelos", filename)
        if os.path.exists(model_path):
            best_model = load_model(model_path)
            metrics = evaluation.avaliar_modelo(best_model, x_test, y_test, name, is_keras_model=True)
            all_metrics.append(metrics)
        else:
            LOGGER.warning(f"Modelo Keras {name} não encontrado. Pule a avaliação ou execute com --retrain.")

    return all_metrics


def compare_train_test_metrics(loaded_classic_models, x_train, y_train, x_test, y_test, tf_available, load_model):
    import gerar_graficos as gg

    train_metrics = []
    test_metrics = []

    for name, model in loaded_classic_models.items():
        try:
            m_train = evaluation.avaliar_modelo(model, x_train, y_train, f"{name}_train", is_keras_model=False)
            m_test = evaluation.avaliar_modelo(model, x_test, y_test, f"{name}_test", is_keras_model=False)
            m_train['modelo'] = name
            m_test['modelo'] = name
            train_metrics.append(m_train)
            test_metrics.append(m_test)
        except Exception as e:
            LOGGER.error(f"Falha ao comparar treino/teste para {name}: {e}")

    if tf_available and load_model:
        keras_models = {
            "MLP": "MLP_best.keras",
            "CNN": "CNN_best.keras",
            "Hibrido_CNN_LSTM": "Hibrido_CNN_LSTM_best.keras"
        }

        for name, filename in keras_models.items():
            model_path = os.path.join(RESULTS_DIR, "modelos", filename)
            if not os.path.exists(model_path):
                continue
            try:
                best_model = load_model(model_path)
                m_train = evaluation.avaliar_modelo(best_model, x_train, y_train, f"{name}_train", is_keras_model=True)
                m_test = evaluation.avaliar_modelo(best_model, x_test, y_test, f"{name}_test", is_keras_model=True)
                m_train['modelo'] = name
                m_test['modelo'] = name
                train_metrics.append(m_train)
                test_metrics.append(m_test)
            except Exception as e:
                LOGGER.error(f"Falha ao comparar treino/teste para {name} (Keras): {e}")

    if train_metrics and test_metrics:
        df_train = pd.DataFrame(train_metrics)
        df_test = pd.DataFrame(test_metrics)
        gg.visualizar_comparacao_treino_teste(df_train, df_test)
        LOGGER.info("Comparação Treino vs Teste gerada com sucesso.")
    else:
        LOGGER.warning("Não foi possível gerar comparação Treino vs Teste (sem métricas).")


def plot_training_history():
    import gerar_graficos as gg

    model_names = ["MLP", "CNN", "Hibrido_CNN_LSTM"]

    for model_name in model_names:
        history_path = os.path.join(RESULTS_DIR, "history", f"{model_name}_history.csv")
        if os.path.exists(history_path):
            df_hist = pd.read_csv(history_path)
            gg.visualizar_historico_treinamento(df_hist, model_name)
            LOGGER.info(f"Curvas de histórico geradas para {model_name}.")
        else:
            LOGGER.warning(f"Arquivo de histórico não encontrado: {history_path}")


def run_pipeline(retrain_models):
    criar_diretorios_projeto()
    LOGGER.info("--- INICIANDO PIPELINE DE PREDIÇÃO DE DIABETES ---")

    df = data_processing.carregar_dados(DATASET_PATH)
    if df is None:
        LOGGER.error("Falha ao carregar os dados. Abortando o pipeline.")
        return

    df = data_processing.analisar_dados(df)
    x_train, x_val, x_test, y_train, y_val, y_test, scaler, encoder, feature_names = data_processing.pre_processar_dados(df)

    tf_available, load_model, modeling, training = check_tensorflow_availability()

    if retrain_models:
        LOGGER.info("--- FASE DE TREINAMENTO (Flag --retrain ativada) ---")
        train_classic_models(x_train, y_train)

        if tf_available:
            train_deep_learning_models(modeling, training, x_train, y_train, x_val, y_val)
        else:
            LOGGER.info("Treinamento de modelos de deep learning ignorado (TensorFlow ausente).")
    else:
        LOGGER.info("--- FASE DE TREINAMENTO PULADA (Usando modelos pré-treinados) ---")

    LOGGER.info("--- FASE DE AVALIAÇÃO ---")

    loaded_classic_models = load_classic_models(x_train, y_train)
    all_metrics = evaluate_classic_models(loaded_classic_models, x_test, y_test)

    if tf_available and load_model:
        keras_metrics = evaluate_keras_models(load_model, x_test, y_test)
        all_metrics.extend(keras_metrics)
    else:
        LOGGER.info("Avaliação de modelos Keras ignorada (TensorFlow indisponível).")

    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        evaluation.comparar_todos_modelos(metrics_df)
    else:
        LOGGER.error("Nenhuma métrica foi gerada. Execute com a flag --retrain primeiro.")

    compare_train_test_metrics(loaded_classic_models, x_train, y_train, x_test, y_test, tf_available, load_model)

    plot_training_history()

    LOGGER.info("--- PIPELINE CONCLUÍDO ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de Treinamento e Avaliação para Predição de Diabetes.")
    parser.add_argument('--retrain', action='store_true', help="Força retreinamento de todos os modelos.")
    args = parser.parse_args()

    run_pipeline(retrain_models=args.retrain)
