import os
import pandas as pd
from config import RESULTS_DIR, LOGGER
import evaluation
import gerar_graficos as gg


def compare_train_test_metrics(loaded_classic_models, x_train, y_train, x_test, y_test, tf_available, load_model):
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
        _add_mlp_metrics(train_metrics, test_metrics, x_train, y_train, x_test, y_test, load_model)
        _add_cnn_metrics(train_metrics, test_metrics, x_train, y_train, x_test, y_test, load_model)

    if train_metrics and test_metrics:
        df_train = pd.DataFrame(train_metrics)
        df_test = pd.DataFrame(test_metrics)
        gg.visualizar_comparacao_treino_teste(df_train, df_test)
        LOGGER.info("Comparação Treino vs Teste gerada com sucesso.")
    else:
        LOGGER.warning("Não foi possível gerar comparação Treino vs Teste (sem métricas).")


def _add_mlp_metrics(train_metrics, test_metrics, x_train, y_train, x_test, y_test, load_model):
    tuned_path = os.path.join(RESULTS_DIR, "modelos", "MLP_Tuned_Final.keras")
    bayesian_path = os.path.join(RESULTS_DIR, "modelos", "MLP_Bayesian_Final.keras")

    if os.path.exists(tuned_path):
        try:
            model = load_model(tuned_path)
            m_train = evaluation.avaliar_modelo(model, x_train, y_train, "MLP_Tuned_train", is_keras_model=True)
            m_test = evaluation.avaliar_modelo(model, x_test, y_test, "MLP_Tuned_test", is_keras_model=True)
            m_train['modelo'] = "MLP_Tuned"
            m_test['modelo'] = "MLP_Tuned"
            train_metrics.append(m_train)
            test_metrics.append(m_test)
        except Exception as e:
            LOGGER.error(f"Falha ao comparar treino/teste para MLP_Tuned: {e}")

    if os.path.exists(bayesian_path):
        try:
            model = load_model(bayesian_path)
            m_train = evaluation.avaliar_modelo(model, x_train, y_train, "MLP_Bayesian_train", is_keras_model=True)
            m_test = evaluation.avaliar_modelo(model, x_test, y_test, "MLP_Bayesian_test", is_keras_model=True)
            m_train['modelo'] = "MLP_Bayesian"
            m_test['modelo'] = "MLP_Bayesian"
            train_metrics.append(m_train)
            test_metrics.append(m_test)
        except Exception as e:
            LOGGER.error(f"Falha ao comparar treino/teste para MLP_Bayesian: {e}")


def _add_cnn_metrics(train_metrics, test_metrics, x_train, y_train, x_test, y_test, load_model):
    tuned_path = os.path.join(RESULTS_DIR, "modelos", "CNN_Tuned_Final.keras")
    bayesian_path = os.path.join(RESULTS_DIR, "modelos", "CNN_Bayesian_Final.keras")

    if os.path.exists(tuned_path):
        try:
            model = load_model(tuned_path)
            m_train = evaluation.avaliar_modelo(model, x_train, y_train, "CNN_Tuned_train", is_keras_model=True)
            m_test = evaluation.avaliar_modelo(model, x_test, y_test, "CNN_Tuned_test", is_keras_model=True)
            m_train['modelo'] = "CNN_Tuned"
            m_test['modelo'] = "CNN_Tuned"
            train_metrics.append(m_train)
            test_metrics.append(m_test)
        except Exception as e:
            LOGGER.error(f"Falha ao comparar treino/teste para CNN_Tuned: {e}")

    if os.path.exists(bayesian_path):
        try:
            model = load_model(bayesian_path)
            m_train = evaluation.avaliar_modelo(model, x_train, y_train, "CNN_Bayesian_train", is_keras_model=True)
            m_test = evaluation.avaliar_modelo(model, x_test, y_test, "CNN_Bayesian_test", is_keras_model=True)
            m_train['modelo'] = "CNN_Bayesian"
            m_test['modelo'] = "CNN_Bayesian"
            train_metrics.append(m_train)
            test_metrics.append(m_test)
        except Exception as e:
            LOGGER.error(f"Falha ao comparar treino/teste para CNN_Bayesian: {e}")
