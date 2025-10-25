import time
import numpy as np
from config import LOGGER
from mlp_utils import check_tensorflow_availability, retrain_final_model


def run_hyperparameter_tuning(x_train, y_train, x_val, y_val, max_trials=50):
    LOGGER.info("=" * 80)
    LOGGER.info("FASE DE OTIMIZAÇÃO DE HIPERPARÂMETROS")
    LOGGER.info("=" * 80)
    LOGGER.info(f"Configuração da busca:")
    LOGGER.info(f"  - Total de trials: {max_trials}")
    LOGGER.info(f"  - Execuções por trial: 2")
    LOGGER.info(f"  - Épocas máximas por trial: 100")
    LOGGER.info(f"  - Objetivo: Maximizar val_precision")
    LOGGER.info(f"\nDados de entrada:")
    LOGGER.info(f"  - Train shape: {x_train.shape}")
    LOGGER.info(f"  - Validation shape: {x_val.shape}")

    start_time = time.time()

    tf_available, _, _, _, hyperparameter_tuning, _ = check_tensorflow_availability()

    if not tf_available:
        LOGGER.error("TensorFlow não disponível. Não é possível realizar tuning.")
        return None, None

    LOGGER.info("\nIniciando busca de hiperparâmetros...")
    LOGGER.info("-" * 80)

    best_model, best_hps, tuner = hyperparameter_tuning.tune_mlp_hyperparameters(
        x_train, y_train, x_val, y_val,
        max_trials=max_trials,
        executions_per_trial=2
    )

    elapsed_time = time.time() - start_time
    LOGGER.info("-" * 80)
    LOGGER.info(f"Busca de hiperparâmetros concluída em {elapsed_time/60:.2f} minutos")

    hyperparameter_tuning.analyze_tuning_results(tuner, top_n=10)

    x_train_full = np.vstack([x_train, x_val])
    y_train_full = np.concatenate([y_train, y_val])

    final_model = retrain_final_model(best_hps, x_train_full, y_train_full, "MLP_Tuned_Final")

    total_time = time.time() - start_time
    LOGGER.info(f"\nTempo total de otimização: {total_time/60:.2f} minutos")

    return final_model, best_hps


def run_bayesian_tuning(x_train, y_train, x_val, y_val, max_trials=30):
    LOGGER.info("=" * 80)
    LOGGER.info("OTIMIZAÇÃO BAYESIANA DE HIPERPARÂMETROS")
    LOGGER.info("=" * 80)

    start_time = time.time()

    tf_available, _, _, _, _, bayesian_tuning = check_tensorflow_availability()

    if not tf_available:
        LOGGER.error("TensorFlow não disponível. Não é possível realizar tuning.")
        return None, None

    best_model, best_hps, tuner = bayesian_tuning.bayesian_tune_mlp(
        x_train, y_train, x_val, y_val,
        max_trials=max_trials,
        executions_per_trial=2
    )

    x_train_full = np.vstack([x_train, x_val])
    y_train_full = np.concatenate([y_train, y_val])

    final_model = retrain_final_model(best_hps, x_train_full, y_train_full, "MLP_Bayesian_Final")

    total_time = time.time() - start_time
    LOGGER.info(f"\nTempo total de otimização bayesiana: {total_time/60:.2f} minutos")

    return final_model, best_hps


def run_cnn_hyperparameter_tuning(x_train, y_train, x_val, y_val, max_trials=50):
    LOGGER.info("=" * 80)
    LOGGER.info("FASE DE OTIMIZAÇÃO DE HIPERPARÂMETROS - CNN")
    LOGGER.info("=" * 80)
    LOGGER.info(f"Configuração da busca:")
    LOGGER.info(f"  - Total de trials: {max_trials}")
    LOGGER.info(f"  - Execuções por trial: 2")
    LOGGER.info(f"  - Épocas máximas por trial: 100")
    LOGGER.info(f"  - Objetivo: Maximizar val_precision")
    LOGGER.info(f"\nDados de entrada:")
    LOGGER.info(f"  - Train shape: {x_train.shape}")
    LOGGER.info(f"  - Validation shape: {x_val.shape}")

    start_time = time.time()

    tf_available, _, _, _, _, _ = check_tensorflow_availability()

    if not tf_available:
        LOGGER.error("TensorFlow não disponível. Não é possível realizar tuning.")
        return None, None

    import cnn_tuning

    LOGGER.info("\nIniciando busca de hiperparâmetros para CNN...")
    LOGGER.info("-" * 80)

    best_model, best_hps, tuner = cnn_tuning.tune_cnn_hyperparameters(
        x_train, y_train, x_val, y_val,
        max_trials=max_trials,
        executions_per_trial=2
    )

    elapsed_time = time.time() - start_time
    LOGGER.info("-" * 80)
    LOGGER.info(f"Busca de hiperparâmetros da CNN concluída em {elapsed_time/60:.2f} minutos")

    cnn_tuning.analyze_cnn_tuning_results(tuner, top_n=10)

    x_train_full = np.vstack([x_train, x_val])
    y_train_full = np.concatenate([y_train, y_val])

    from cnn_utils import retrain_final_cnn
    final_model = retrain_final_cnn(best_hps, x_train_full, y_train_full, "CNN_Tuned_Final")

    total_time = time.time() - start_time
    LOGGER.info(f"\nTempo total de otimização da CNN: {total_time/60:.2f} minutos")

    return final_model, best_hps


def run_cnn_bayesian_tuning(x_train, y_train, x_val, y_val, max_trials=30):
    LOGGER.info("=" * 80)
    LOGGER.info("OTIMIZAÇÃO BAYESIANA DE HIPERPARÂMETROS - CNN")
    LOGGER.info("=" * 80)

    start_time = time.time()

    tf_available, _, _, _, _, _ = check_tensorflow_availability()

    if not tf_available:
        LOGGER.error("TensorFlow não disponível. Não é possível realizar tuning.")
        return None, None

    import cnn_tuning

    best_model, best_hps, tuner = cnn_tuning.bayesian_tune_cnn(
        x_train, y_train, x_val, y_val,
        max_trials=max_trials,
        executions_per_trial=2
    )

    x_train_full = np.vstack([x_train, x_val])
    y_train_full = np.concatenate([y_train, y_val])

    from cnn_utils import retrain_final_cnn
    final_model = retrain_final_cnn(best_hps, x_train_full, y_train_full, "CNN_Bayesian_Final")

    total_time = time.time() - start_time
    LOGGER.info(f"\nTempo total de otimização bayesiana da CNN: {total_time/60:.2f} minutos")

    return final_model, best_hps
