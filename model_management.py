import os
import pickle
from config import RESULTS_DIR, LOGGER
from utils import RANDOM_STATE
import evaluation


def train_classic_models(x_train, y_train):
    from modeling import obter_modelos_classicos
    from training import treinar_modelos_classicos_pt

    LOGGER.info("Treinando modelos clássicos...")
    classic_models = obter_modelos_classicos(RANDOM_STATE)
    treinar_modelos_classicos_pt(classic_models, x_train, y_train)


def load_classic_models(x_train, y_train):
    from modeling import obter_modelos_classicos

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


def evaluate_mlp_models(x_test, y_test, tf_available, load_model):
    if not tf_available or not load_model:
        return []

    all_metrics = []
    tuned_final_path = os.path.join(RESULTS_DIR, "modelos", "MLP_Tuned_Final.keras")
    tuned_best_path = os.path.join(RESULTS_DIR, "modelos", "MLP_Tuned_best.keras")
    bayesian_path = os.path.join(RESULTS_DIR, "modelos", "MLP_Bayesian_Final.keras")

    if os.path.exists(bayesian_path):
        LOGGER.info("Avaliando modelo MLP Bayesian...")
        model = load_model(bayesian_path)
        metrics = evaluation.avaliar_modelo(model, x_test, y_test, "MLP_Bayesian", is_keras_model=True)
        all_metrics.append(metrics)
    elif os.path.exists(tuned_final_path):
        LOGGER.info("Avaliando modelo MLP Tuned (Final)...")
        model = load_model(tuned_final_path)
        metrics = evaluation.avaliar_modelo(model, x_test, y_test, "MLP_Tuned", is_keras_model=True)
        all_metrics.append(metrics)
    elif os.path.exists(tuned_best_path):
        LOGGER.info("Avaliando modelo MLP Tuned (Best)...")
        model = load_model(tuned_best_path)
        metrics = evaluation.avaliar_modelo(model, x_test, y_test, "MLP_Tuned", is_keras_model=True)
        all_metrics.append(metrics)

    return all_metrics


def evaluate_cnn_models(x_test, y_test, tf_available, load_model):
    if not tf_available or not load_model:
        return []

    all_metrics = []
    tuned_final_path = os.path.join(RESULTS_DIR, "modelos", "CNN_Tuned_Final.keras")
    tuned_best_path = os.path.join(RESULTS_DIR, "modelos", "CNN_Tuned_best.keras")
    bayesian_path = os.path.join(RESULTS_DIR, "modelos", "CNN_Bayesian_Final.keras")

    if os.path.exists(bayesian_path):
        LOGGER.info("Avaliando modelo CNN Bayesian...")
        model = load_model(bayesian_path)
        metrics = evaluation.avaliar_modelo(model, x_test, y_test, "CNN_Bayesian", is_keras_model=True)
        all_metrics.append(metrics)
    elif os.path.exists(tuned_final_path):
        LOGGER.info("Avaliando modelo CNN Tuned (Final)...")
        model = load_model(tuned_final_path)
        metrics = evaluation.avaliar_modelo(model, x_test, y_test, "CNN_Tuned", is_keras_model=True)
        all_metrics.append(metrics)
    elif os.path.exists(tuned_best_path):
        LOGGER.info("Avaliando modelo CNN Tuned (Best)...")
        model = load_model(tuned_best_path)
        metrics = evaluation.avaliar_modelo(model, x_test, y_test, "CNN_Tuned", is_keras_model=True)
        all_metrics.append(metrics)

    return all_metrics
