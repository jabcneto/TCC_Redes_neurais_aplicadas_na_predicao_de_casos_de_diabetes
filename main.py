import argparse
import pandas as pd

from config import LOGGER, criar_diretorios_projeto
from utils import DATASET_PATH
import data_processing
import evaluation

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYCARET_CUSTOM_LOGGING_LEVEL'] = 'CRITICAL'

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

from mlp_utils import check_tensorflow_availability
from model_management import load_classic_models, evaluate_classic_models, evaluate_mlp_models, evaluate_cnn_models
from comparison_utils import compare_train_test_metrics
from tuning_pipelines import (
    run_hyperparameter_tuning,
    run_bayesian_tuning,
    run_cnn_hyperparameter_tuning,
    run_cnn_bayesian_tuning
)
from cv_pipelines import (
    run_nested_cross_validation,
    run_cross_validation_with_pretrained,
    run_cross_validation_after_tuning
)


def run_evaluation_pipeline(x_train, y_train, x_test, y_test):
    LOGGER.info("\n--- FASE DE AVALIAÇÃO ---")

    loaded_classic_models = load_classic_models(x_train, y_train)
    all_metrics = evaluate_classic_models(loaded_classic_models, x_test, y_test)

    tf_available, load_model, *_ = check_tensorflow_availability()
    mlp_metrics = evaluate_mlp_models(x_test, y_test, tf_available, load_model)
    cnn_metrics = evaluate_cnn_models(x_test, y_test, tf_available, load_model)
    all_metrics.extend(mlp_metrics)
    all_metrics.extend(cnn_metrics)

    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        evaluation.comparar_todos_modelos(metrics_df)
        compare_train_test_metrics(loaded_classic_models, x_train, y_train, x_test, y_test, tf_available, load_model)
    else:
        LOGGER.error("Nenhuma métrica foi gerada. Execute com --tune ou --bayesian primeiro.")


def run_tuning_pipeline(x_train, y_train, x_val, y_val, x_test, y_test, use_bayesian, use_cv, n_folds, tuning_trials):
    if use_bayesian:
        best_model, best_hps = run_bayesian_tuning(x_train, y_train, x_val, y_val, max_trials=tuning_trials)
    else:
        best_model, best_hps = run_hyperparameter_tuning(x_train, y_train, x_val, y_val, max_trials=tuning_trials)

    if best_model is not None:
        LOGGER.info("\n--- AVALIAÇÃO DO MODELO OTIMIZADO ---")
        tuned_metrics = evaluation.avaliar_modelo(best_model, x_test, y_test, "MLP_Tuned", is_keras_model=True)
        LOGGER.info(f"Precision: {tuned_metrics['precision']:.4f}")
        LOGGER.info(f"Recall: {tuned_metrics['recall']:.4f}")
        LOGGER.info(f"F1-Score: {tuned_metrics['f1']:.4f}")
        LOGGER.info(f"AUC-ROC: {tuned_metrics['roc_auc']:.4f}")

        if use_cv:
            run_cross_validation_after_tuning(best_hps, tuned_metrics, x_train, y_train, x_val, y_val, n_folds)


def run_cnn_tuning_pipeline(x_train, y_train, x_val, y_val, x_test, y_test, use_bayesian, use_cv, n_folds, tuning_trials):
    if use_bayesian:
        best_model, best_hps = run_cnn_bayesian_tuning(x_train, y_train, x_val, y_val, max_trials=tuning_trials)
    else:
        best_model, best_hps = run_cnn_hyperparameter_tuning(x_train, y_train, x_val, y_val, max_trials=tuning_trials)

    if best_model is not None:
        LOGGER.info("\n--- AVALIAÇÃO DO MODELO CNN OTIMIZADO ---")
        tuned_metrics = evaluation.avaliar_modelo(best_model, x_test, y_test, "CNN_Tuned", is_keras_model=True)
        LOGGER.info(f"Precision: {tuned_metrics['precision']:.4f}")
        LOGGER.info(f"Recall: {tuned_metrics['recall']:.4f}")
        LOGGER.info(f"F1-Score: {tuned_metrics['f1']:.4f}")
        LOGGER.info(f"AUC-ROC: {tuned_metrics['roc_auc']:.4f}")

        if use_cv:
            run_cross_validation_after_tuning(best_hps, tuned_metrics, x_train, y_train, x_val, y_val, n_folds)


def run_pipeline(tune_hyperparameters=False, use_bayesian=False, tune_cnn=False, use_bayesian_cnn=False,
                 use_cv=False, use_nested_cv=False, n_folds=5, tuning_trials=50,
                 balance_strategy='smote', sampling_strategy=0.7, k_neighbors=5, auto_balance=False):
    criar_diretorios_projeto()
    LOGGER.info("--- INICIANDO PIPELINE DE PREDIÇÃO DE DIABETES ---")

    df = data_processing.carregar_dados(DATASET_PATH)
    if df is None:
        LOGGER.error("Falha ao carregar os dados. Abortando o pipeline.")
        return

    df = data_processing.analisar_dados(df)
    x_train, x_val, x_test, y_train, y_val, y_test, scaler, encoder, feature_names = data_processing.pre_processar_dados(
        df,
        balance_strategy=balance_strategy,
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        auto_balance=auto_balance
    )

    if use_nested_cv:
        run_nested_cross_validation(x_train, y_train, x_val, y_val, n_folds, tuning_trials)
        return

    if use_cv and not tune_hyperparameters and not tune_cnn:
        tf_available, load_model, *_ = check_tensorflow_availability()
        if tf_available:
            run_cross_validation_with_pretrained(x_train, y_train, x_val, y_val, x_test, y_test, n_folds, load_model)
        return

    if tune_cnn:
        run_cnn_tuning_pipeline(x_train, y_train, x_val, y_val, x_test, y_test, use_bayesian_cnn, use_cv, n_folds, tuning_trials)
        return

    if tune_hyperparameters:
        run_tuning_pipeline(x_train, y_train, x_val, y_val, x_test, y_test, use_bayesian, use_cv, n_folds, tuning_trials)
        return

    run_evaluation_pipeline(x_train, y_train, x_test, y_test)
    LOGGER.info("\n--- PIPELINE CONCLUÍDO ---")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Pipeline de Treinamento e Avaliação para Predição de Diabetes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

  MLP:
  1. Buscar melhores hiperparâmetros MLP (50 trials):
     python3 main.py --tune --trials 50

  2. Buscar hiperparâmetros MLP com Bayesian (30 trials):
     python3 main.py --bayesian --trials 30

  CNN:
  3. Buscar melhores hiperparâmetros CNN (50 trials):
     python3 main.py --tune-cnn --trials 50

  4. Buscar hiperparâmetros CNN com Bayesian (30 trials):
     python3 main.py --bayesian-cnn --trials 30

  Validação:
  5. Validação cruzada com modelo pré-treinado:
     python3 main.py --cv

  6. Validação cruzada aninhada:
     python3 main.py --nested-cv --folds 5

  7. Apenas avaliar modelos já treinados:
     python3 main.py
        """
    )

    parser.add_argument(
        '--tune',
        action='store_true',
        help="Busca intensiva de hiperparâmetros MLP para máxima precisão (pode levar horas)."
    )

    parser.add_argument(
        '--bayesian',
        action='store_true',
        help="Busca bayesiana de hiperparâmetros MLP (pode ser mais rápida e eficiente)."
    )

    parser.add_argument(
        '--tune-cnn',
        action='store_true',
        help="Busca intensiva de hiperparâmetros CNN para máxima precisão (pode levar horas)."
    )

    parser.add_argument(
        '--bayesian-cnn',
        action='store_true',
        help="Busca bayesiana de hiperparâmetros CNN (pode ser mais rápida e eficiente)."
    )

    parser.add_argument(
        '--cv',
        action='store_true',
        help="Realiza validação cruzada com K-Folds (5 folds) após o tuning."
    )

    parser.add_argument(
        '--nested-cv',
        action='store_true',
        help="Validação cruzada aninhada para estimativa de generalização não enviesada."
    )

    parser.add_argument(
        '--folds',
        type=int,
        default=5,
        help="Número de folds para validação cruzada (padrão: 5)."
    )

    parser.add_argument(
        '--trials',
        type=int,
        default=50,
        help="Número de trials para tuning. Sugestões: 15 (rápido), 50 (padrão), 100 (intensivo)."
    )

    parser.add_argument(
        '--balance-strategy',
        type=str,
        default='smote',
        choices=['smote', 'smotenc', 'adasyn', 'smote_tomek', 'smote_enn', 'ros', 'rus', 'none'],
        help="Estratégia de balanceamento: smote, smotenc, adasyn, smote_tomek, smote_enn, ros, rus, none. Padrão: smote."
    )

    parser.add_argument(
        '--sampling-strategy',
        type=float,
        default=0.7,
        help="Proporção alvo da classe minoritária após reamostragem (ex.: 0.4). Padrão: 0.7."
    )

    parser.add_argument(
        '--k-neighbors',
        type=int,
        default=5,
        help="Número de vizinhos para métodos baseados em KNN (SMOTE/ADASYN). Padrão: 5."
    )

    parser.add_argument(
        '--auto-balance',
        action='store_true',
        help="Seleciona automaticamente a melhor estratégia de balanceamento com base na precisão em validação."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    run_pipeline(
        tune_hyperparameters=args.tune or args.bayesian,
        use_bayesian=args.bayesian,
        tune_cnn=args.tune_cnn or args.bayesian_cnn,
        use_bayesian_cnn=args.bayesian_cnn,
        use_cv=args.cv,
        use_nested_cv=args.nested_cv,
        n_folds=args.folds,
        tuning_trials=args.trials,
        balance_strategy=args.balance_strategy,
        sampling_strategy=args.sampling_strategy,
        k_neighbors=args.k_neighbors,
        auto_balance=args.auto_balance
    )
