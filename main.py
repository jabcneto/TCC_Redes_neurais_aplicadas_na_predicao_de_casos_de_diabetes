import argparse
import pandas as pd
import os

from config import LOGGER, criar_diretorios_projeto
from utils import DATASET_PATH
import data_processing
import evaluation

import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

from mlp_utils import check_tensorflow_availability
from model_management import load_classic_models, evaluate_classic_models, evaluate_mlp_models, evaluate_cnn_models
from comparison_utils import compare_train_test_metrics
from tuning_pipelines import (
    run_bayesian_tuning,
    run_cnn_bayesian_tuning
)
from consolidate_tuning import consolidate_tuning

from training import treinar_modelo_keras_pt
from config import DEFAULT_FINAL_TRAINING_EPOCHS, DEFAULT_BATCH_SIZE, RESULTS_DIR
from bayesian_tuning import load_hps_from_results as load_mlp_hps_from_results, create_mlp_from_hps
from cnn_tuning import load_cnn_hps_from_trial_json, create_cnn_from_hps


def _fmt4(v):
    try:
        return f"{float(v):.4f}"
    except Exception:
        return "NA"


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
        LOGGER.error("Nenhuma métrica foi gerada. Execute com --bayesian primeiro.")


def run_tuning_pipeline(x_train, y_train, x_val, y_val, x_test, y_test, tuning_trials):
    best_model, best_hps = run_bayesian_tuning(x_train, y_train, x_val, y_val, max_trials=tuning_trials)

    if best_model is not None:
        LOGGER.info("\n--- AVALIAÇÃO DO MODELO OTIMIZADO ---")
        tuned_metrics = evaluation.avaliar_modelo(best_model, x_test, y_test, "MLP_Tuned", is_keras_model=True)
        LOGGER.info(f"Precision: {_fmt4(tuned_metrics.get('precision'))}")
        LOGGER.info(f"Recall: {_fmt4(tuned_metrics.get('recall'))}")
        LOGGER.info(f"F1-Score: {_fmt4(tuned_metrics.get('f1'))}")
        LOGGER.info(f"AUC-ROC: {_fmt4(tuned_metrics.get('roc_auc'))}")


def run_cnn_tuning_pipeline(x_train, y_train, x_val, y_val, x_test, y_test, tuning_trials):
    best_model, best_hps = run_cnn_bayesian_tuning(x_train, y_train, x_val, y_val, max_trials=tuning_trials)

    if best_model is not None:
        LOGGER.info("\n--- AVALIAÇÃO DO MODELO CNN OTIMIZADO ---")
        tuned_metrics = evaluation.avaliar_modelo(best_model, x_test, y_test, "CNN_Tuned", is_keras_model=True)
        LOGGER.info(f"Precision: {_fmt4(tuned_metrics.get('precision'))}")
        LOGGER.info(f"Recall: {_fmt4(tuned_metrics.get('recall'))}")
        LOGGER.info(f"F1-Score: {_fmt4(tuned_metrics.get('f1'))}")
        LOGGER.info(f"AUC-ROC: {_fmt4(tuned_metrics.get('roc_auc'))}")


def run_pipeline(tune_hyperparameters=False,
                 tuning_trials=50,
                 train_mlp_trial_number=None,
                 train_mlp_trial_id=None,
                 train_cnn_trial_json=None,
                 run_mlp_bayesian_flag=False,
                 run_cnn_bayesian_flag=False):
    criar_diretorios_projeto()
    LOGGER.info("--- INICIANDO PIPELINE DE PREDIÇÃO DE DIABETES ---")

    df = data_processing.carregar_dados(DATASET_PATH)
    if df is None:
        LOGGER.error("Falha ao carregar os dados. Abortando o pipeline.")
        return

    df = data_processing.analisar_dados(df)
    (x_train,
     x_val,
     x_test,
     y_train,
     y_val,
     y_test,
     scaler,
     encoder,
     feature_names
     ) = data_processing.pre_processar_dados(df)

    if train_mlp_trial_number is not None or train_mlp_trial_id is not None:
        results_dir = os.path.join(RESULTS_DIR, 'tuning', 'bayesian_results')
        hp = load_mlp_hps_from_results(results_dir, trial_number=train_mlp_trial_number, trial_id=train_mlp_trial_id)
        if hp is None:
            LOGGER.error("Hiperparâmetros do MLP não encontrados para o trial especificado.")
            return
        hp['input_dim'] = x_train.shape[1]
        model = create_mlp_from_hps(hp)
        batch_size = hp.get('batch_size', DEFAULT_BATCH_SIZE)
        epochs = DEFAULT_FINAL_TRAINING_EPOCHS
        model, history = treinar_modelo_keras_pt(model, x_train, y_train, x_val, y_val, nome_modelo="MLP_Bayesian_Selected", epochs=epochs, batch_size=batch_size)
        tuned_metrics = evaluation.avaliar_modelo(model, x_test, y_test, "MLP_Selected", is_keras_model=True)
        LOGGER.info(f"Precision: {_fmt4(tuned_metrics.get('precision'))}")
        LOGGER.info(f"Recall: {_fmt4(tuned_metrics.get('recall'))}")
        LOGGER.info(f"F1-Score: {_fmt4(tuned_metrics.get('f1'))}")
        LOGGER.info(f"AUC-ROC: {_fmt4(tuned_metrics.get('roc_auc'))}")
        LOGGER.info("\n--- PIPELINE CONCLUÍDO ---")
        return

    if train_cnn_trial_json is not None:
        hps = load_cnn_hps_from_trial_json(train_cnn_trial_json)
        if hps is None:
            LOGGER.error("Hiperparâmetros da CNN não encontrados no arquivo informado.")
            return
        model = create_cnn_from_hps(hps, input_dim=x_train.shape[1])
        batch_size = hps.get('batch_size', DEFAULT_BATCH_SIZE)
        epochs = DEFAULT_FINAL_TRAINING_EPOCHS
        model, history = treinar_modelo_keras_pt(model, x_train, y_train, x_val, y_val, nome_modelo="CNN_Bayesian_Selected", epochs=epochs, batch_size=batch_size)
        tuned_metrics = evaluation.avaliar_modelo(model, x_test, y_test, "CNN_Selected", is_keras_model=True)
        LOGGER.info(f"Precision: {_fmt4(tuned_metrics.get('precision'))}")
        LOGGER.info(f"Recall: {_fmt4(tuned_metrics.get('recall'))}")
        LOGGER.info(f"F1-Score: {_fmt4(tuned_metrics.get('f1'))}")
        LOGGER.info(f"AUC-ROC: {_fmt4(tuned_metrics.get('roc_auc'))}")
        LOGGER.info("\n--- PIPELINE CONCLUÍDO ---")
        return

    run_mlp = False
    run_cnn = False
    if run_mlp_bayesian_flag or run_cnn_bayesian_flag:
        run_mlp = run_mlp_bayesian_flag
        run_cnn = run_cnn_bayesian_flag
    elif tune_hyperparameters:
        run_mlp = True
        run_cnn = True

    any_ran = False

    if run_cnn:
        run_cnn_tuning_pipeline(x_train, y_train, x_val, y_val, x_test, y_test, tuning_trials)
        any_ran = True

    if run_mlp:
        run_tuning_pipeline(x_train, y_train, x_val, y_val, x_test, y_test, tuning_trials)
        any_ran = True

    if any_ran:
        tuning_root = os.path.join(os.path.dirname(__file__), "resultados_diabetes", "tuning")
        t_csv, e_csv = consolidate_tuning(tuning_root)
        LOGGER.info(f"CSV consolidado de trials: {t_csv}")
        LOGGER.info(f"CSV consolidado de épocas: {e_csv}")
        return

    run_evaluation_pipeline(x_train, y_train, x_test, y_test)
    LOGGER.info("\n--- PIPELINE CONCLUÍDO ---")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Pipeline de Treinamento e Avaliação para Predição de Diabetes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

  Avaliar modelos já treinados:
     python main.py

  Tuning bayesiano (CNN + MLP) com 30 trials:
     python main.py --bayesian --trials 30

  Tuning apenas MLP:
     python main.py --bayesian-mlp --trials 30

  Tuning apenas CNN:
     python main.py --bayesian-cnn --trials 30

  Retreinar MLP por trial number:
     python main.py --train-mlp-trial-number 7

  Retreinar MLP por trial id:
     python main.py --train-mlp-trial-id 0007

  Retreinar CNN a partir de um trial.json do Keras Tuner:
     python main.py --train-cnn-trial-json resultados_diabetes/tuning/cnn_bayesian_results/cnn_bayesian_tuning/trial_00/trial.json
        """
    )

    parser.add_argument(
        '--bayesian',
        action='store_true',
        help="Executa busca bayesiana de hiperparâmetros para CNN e MLP."
    )

    parser.add_argument(
        '--bayesian-mlp',
        action='store_true',
        help="Executa busca bayesiana apenas para o MLP."
    )

    parser.add_argument(
        '--bayesian-cnn',
        action='store_true',
        help="Executa busca bayesiana apenas para a CNN."
    )

    parser.add_argument(
        '--trials',
        type=int,
        default=50,
        help="Número de trials para tuning. Sugestões: 15 (rápido), 50 (padrão), 100 (intensivo)."
    )

    parser.add_argument(
        '--train-mlp-trial-number',
        type=int,
        help="Número do trial do MLP em bayesian_trials_detailed.json para retreino."
    )

    parser.add_argument(
        '--train-mlp-trial-id',
        type=str,
        help="ID do trial do MLP em bayesian_trials_detailed.json para retreino."
    )

    parser.add_argument(
        '--train-cnn-trial-json',
        type=str,
        help="Caminho para trial.json do Keras Tuner para retreino de CNN."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    run_pipeline(
        tune_hyperparameters=args.bayesian,
        tuning_trials=args.trials,
        train_mlp_trial_number=getattr(args, 'train_mlp_trial_number', None),
        train_mlp_trial_id=getattr(args, 'train_mlp_trial_id', None),
        train_cnn_trial_json=getattr(args, 'train_cnn_trial_json', None),
        run_mlp_bayesian_flag=getattr(args, 'bayesian_mlp', False),
        run_cnn_bayesian_flag=getattr(args, 'bayesian_cnn', False)
    )
