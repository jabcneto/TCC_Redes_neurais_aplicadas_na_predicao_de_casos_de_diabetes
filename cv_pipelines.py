import os
import json
import numpy as np
from config import RESULTS_DIR, LOGGER, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE
import cross_validation
import evaluation
from mlp_utils import create_model_from_hyperparameters
from cnn_utils import create_cnn_from_hyperparameters


def run_nested_cross_validation(x_train, y_train, x_val, y_val, n_folds, max_trials, epochs=None):
    if epochs is None:
        epochs = DEFAULT_EPOCHS

    LOGGER.info("\n--- VALIDAÇÃO CRUZADA ANINHADA ---")
    LOGGER.info(f"Epochs por fold: {epochs}")

    x_train_full = np.vstack([x_train, x_val])
    y_train_full = np.concatenate([y_train, y_val])

    def model_builder_func(hp):
        hp.Fixed('input_dim', value=x_train_full.shape[1])
        from hyperparameter_tuning import build_tunable_mlp
        return build_tunable_mlp(hp)

    nested_results = cross_validation.nested_cross_validation(
        model_builder_func=model_builder_func,
        hyperparameter_space={},
        x_train=x_train_full,
        y_train=y_train_full,
        outer_folds=n_folds,
        inner_folds=3,
        max_trials=max_trials,
        epochs=epochs,
        batch_size=DEFAULT_BATCH_SIZE,
        verbose=1
    )
    return nested_results


def run_cross_validation_with_pretrained(x_train, y_train, x_val, y_val, x_test, y_test, n_folds, load_model, epochs=None):
    if epochs is None:
        epochs = DEFAULT_EPOCHS

    LOGGER.info("\n--- VALIDAÇÃO CRUZADA COM MODELO PRÉ-TREINADO ---")
    LOGGER.info(f"Epochs por fold: {epochs}")

    bayesian_config_path = os.path.join(RESULTS_DIR, 'tuning', 'bayesian_results', 'bayesian_best_config.json')
    standard_config_path = os.path.join(RESULTS_DIR, 'tuning', 'results', 'best_trial_config.json')

    config_path = None
    if os.path.exists(bayesian_config_path):
        config_path = bayesian_config_path
        LOGGER.info(f"Usando configuração bayesiana: {bayesian_config_path}")
    elif os.path.exists(standard_config_path):
        config_path = standard_config_path
        LOGGER.info(f"Usando configuração padrão: {standard_config_path}")
    else:
        LOGGER.error("Nenhuma configuração de hiperparâmetros encontrada!")
        LOGGER.error("Execute primeiro: python main.py --tune ou python main.py --bayesian")
        return

    with open(config_path, 'r') as f:
        config = json.load(f)

    best_hps_dict = config['hyperparameters']

    LOGGER.info(f"Carregando hiperparâmetros do trial #{config['trial_number']}")
    LOGGER.info(f"Precisão original: {config['best_val_precision']:.4f}")

    x_train_full = np.vstack([x_train, x_val])
    y_train_full = np.concatenate([y_train, y_val])

    def create_model_from_config():
        return create_model_from_hyperparameters(best_hps_dict, (x_train_full.shape[1],))

    cv_results = cross_validation.cross_validate_mlp(
        model_builder=create_model_from_config,
        x_train=x_train_full,
        y_train=y_train_full,
        n_folds=n_folds,
        epochs=epochs,
        batch_size=best_hps_dict.get('batch_size', DEFAULT_BATCH_SIZE)
    )

    _compare_cv_with_holdout_models(cv_results, x_test, y_test, load_model)
    LOGGER.info("\n--- PIPELINE DE VALIDAÇÃO CRUZADA CONCLUÍDO ---")


def run_cnn_cross_validation_with_pretrained(x_train, y_train, x_val, y_val, x_test, y_test, n_folds, load_model, epochs=None):
    if epochs is None:
        epochs = DEFAULT_EPOCHS

    LOGGER.info("\n--- VALIDAÇÃO CRUZADA COM CNN PRÉ-TREINADA ---")
    LOGGER.info(f"Epochs por fold: {epochs}")

    bayesian_config_path = os.path.join(RESULTS_DIR, 'tuning', 'cnn_bayesian_results', 'bayesian_best_config.json')
    standard_config_path = os.path.join(RESULTS_DIR, 'tuning', 'cnn_results', 'best_trial_config.json')

    config_path = None
    if os.path.exists(bayesian_config_path):
        config_path = bayesian_config_path
        LOGGER.info(f"Usando configuração bayesiana da CNN: {bayesian_config_path}")
    elif os.path.exists(standard_config_path):
        config_path = standard_config_path
        LOGGER.info(f"Usando configuração padrão da CNN: {standard_config_path}")
    else:
        LOGGER.error("Nenhuma configuração de hiperparâmetros CNN encontrada!")
        LOGGER.error("Execute primeiro: python main.py --tune-cnn ou python main.py --bayesian-cnn")
        return

    with open(config_path, 'r') as f:
        config = json.load(f)

    best_hps_dict = config['hyperparameters']

    LOGGER.info(f"Carregando hiperparâmetros do trial #{config['trial_number']}")
    LOGGER.info(f"Precisão original: {config['best_val_precision']:.4f}")

    x_train_full = np.vstack([x_train, x_val])
    y_train_full = np.concatenate([y_train, y_val])

    def create_model_from_config():
        return create_cnn_from_hyperparameters(best_hps_dict, (x_train_full.shape[1],))

    cv_results = cross_validation.cross_validate_mlp(
        model_builder=create_model_from_config,
        x_train=x_train_full,
        y_train=y_train_full,
        n_folds=n_folds,
        epochs=epochs,
        batch_size=best_hps_dict.get('batch_size', DEFAULT_BATCH_SIZE)
    )

    _compare_cnn_cv_with_holdout(cv_results, x_test, y_test, load_model)
    LOGGER.info("\n--- PIPELINE DE VALIDAÇÃO CRUZADA CNN CONCLUÍDO ---")


def _compare_cv_with_holdout_models(cv_results, x_test, y_test, load_model):
    bayesian_model_path = os.path.join(RESULTS_DIR, "modelos", "MLP_Bayesian_Final.keras")
    tuned_model_path = os.path.join(RESULTS_DIR, "modelos", "MLP_Tuned_Final.keras")

    if os.path.exists(bayesian_model_path):
        LOGGER.info("\n--- COMPARANDO CV COM MODELO BAYESIANO ---")
        loaded_model = load_model(bayesian_model_path)
        holdout_metrics = evaluation.avaliar_modelo(loaded_model, x_test, y_test, "Bayesian_Holdout", is_keras_model=True)
        cross_validation.compare_cv_with_holdout(cv_results, holdout_metrics, model_name="MLP_Bayesian")
    elif os.path.exists(tuned_model_path):
        LOGGER.info("\n--- COMPARANDO CV COM MODELO TUNED ---")
        loaded_model = load_model(tuned_model_path)
        holdout_metrics = evaluation.avaliar_modelo(loaded_model, x_test, y_test, "Tuned_Holdout", is_keras_model=True)
        cross_validation.compare_cv_with_holdout(cv_results, holdout_metrics, model_name="MLP_Tuned")


def _compare_cnn_cv_with_holdout(cv_results, x_test, y_test, load_model):
    bayesian_model_path = os.path.join(RESULTS_DIR, "modelos", "CNN_Bayesian_Final.keras")
    tuned_model_path = os.path.join(RESULTS_DIR, "modelos", "CNN_Tuned_Final.keras")

    if os.path.exists(bayesian_model_path):
        LOGGER.info("\n--- COMPARANDO CV COM MODELO CNN BAYESIANO ---")
        loaded_model = load_model(bayesian_model_path)
        holdout_metrics = evaluation.avaliar_modelo(loaded_model, x_test, y_test, "CNN_Bayesian_Holdout", is_keras_model=True)
        cross_validation.compare_cv_with_holdout(cv_results, holdout_metrics, model_name="CNN_Bayesian")
    elif os.path.exists(tuned_model_path):
        LOGGER.info("\n--- COMPARANDO CV COM MODELO CNN TUNED ---")
        loaded_model = load_model(tuned_model_path)
        holdout_metrics = evaluation.avaliar_modelo(loaded_model, x_test, y_test, "CNN_Tuned_Holdout", is_keras_model=True)
        cross_validation.compare_cv_with_holdout(cv_results, holdout_metrics, model_name="CNN_Tuned")


def run_cross_validation_after_tuning(best_hps, tuned_metrics, x_train, y_train, x_val, y_val, n_folds, epochs=None):
    if epochs is None:
        epochs = DEFAULT_EPOCHS

    LOGGER.info("\n--- VALIDAÇÃO CRUZADA COM MELHORES HIPERPARÂMETROS ---")
    LOGGER.info(f"Epochs por fold: {epochs}")

    x_train_full = np.vstack([x_train, x_val])
    y_train_full = np.concatenate([y_train, y_val])

    cv_results = cross_validation.cross_validate_with_tuning(
        model_builder_func=None,
        best_hps=best_hps,
        x_train=x_train_full,
        y_train=y_train_full,
        n_folds=n_folds,
        epochs=epochs,
        batch_size=DEFAULT_BATCH_SIZE
    )

    if hasattr(best_hps, 'values'):
        hps_values = best_hps.values
    else:
        hps_values = best_hps

    if 'num_conv_layers' in hps_values:
        model_name = "CNN_Tuned"
    else:
        model_name = "MLP_Tuned"

    cross_validation.compare_cv_with_holdout(cv_results, tuned_metrics, model_name=model_name)
