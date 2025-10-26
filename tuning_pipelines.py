import time
import numpy as np
from config import LOGGER, DEFAULT_TUNING_EPOCHS
from mlp_utils import check_tensorflow_availability, retrain_final_model
from progress_callback import ProgressCallback

def run_bayesian_tuning(x_train, y_train, x_val, y_val, max_trials=30, epochs=None):
    if epochs is None:
        epochs = DEFAULT_TUNING_EPOCHS

    LOGGER.info("=" * 80)
    LOGGER.info("OTIMIZAÇÃO BAYESIANA DE HIPERPARÂMETROS")
    LOGGER.info("=" * 80)
    LOGGER.info(f"Total de trials: {max_trials}")
    LOGGER.info(f"Épocas por trial: {epochs}")
    LOGGER.info(f"Dados - Train: {x_train.shape}, Validation: {x_val.shape}")
    LOGGER.info("=" * 80 + "\n")

    start_time = time.time()

    tf_available, _, _, _, _, bayesian_tuning = check_tensorflow_availability()

    if not tf_available:
        LOGGER.error("TensorFlow não disponível. Não é possível realizar tuning.")
        return None, None

    best_model, best_hps, tuner = bayesian_tuning.bayesian_tune_mlp(
        x_train, y_train, x_val, y_val,
        max_trials=max_trials,
        executions_per_trial=2,
        progress_callback=None,
        epochs=epochs
    )

    x_train_full = np.vstack([x_train, x_val])
    y_train_full = np.concatenate([y_train, y_val])

    final_model = retrain_final_model(best_hps, x_train_full, y_train_full, "MLP_Bayesian_Final")

    total_time = time.time() - start_time
    LOGGER.info(f"\nTempo total de otimização bayesiana: {total_time/60:.2f} minutos")

    return final_model, best_hps

def run_cnn_bayesian_tuning(x_train, y_train, x_val, y_val, max_trials=30, epochs=None):
    if epochs is None:
        epochs = DEFAULT_TUNING_EPOCHS

    LOGGER.info("=" * 80)
    LOGGER.info("OTIMIZAÇÃO BAYESIANA DE HIPERPARÂMETROS - CNN")
    LOGGER.info("=" * 80)
    LOGGER.info(f"Épocas por trial: {epochs}")

    start_time = time.time()

    tf_available, _, _, _, _, _ = check_tensorflow_availability()

    if not tf_available:
        LOGGER.error("TensorFlow não disponível. Não é possível realizar tuning.")
        return None, None

    import cnn_tuning

    progress_tracker = ProgressCallback(max_trials, "CNN Bayesian")

    best_model, best_hps, tuner = cnn_tuning.bayesian_tune_cnn(
        x_train, y_train, x_val, y_val,
        max_trials=max_trials,
        executions_per_trial=2,
        progress_callback=progress_tracker,
        epochs=epochs
    )

    x_train_full = np.vstack([x_train, x_val])
    y_train_full = np.concatenate([y_train, y_val])

    from cnn_utils import retrain_final_cnn
    final_model = retrain_final_cnn(best_hps, x_train_full, y_train_full, "CNN_Bayesian_Final")

    total_time = time.time() - start_time
    progress_tracker.print_final_summary()
    LOGGER.info(f"\nTempo total de otimização bayesiana da CNN: {total_time/60:.2f} minutos")

    return final_model, best_hps
