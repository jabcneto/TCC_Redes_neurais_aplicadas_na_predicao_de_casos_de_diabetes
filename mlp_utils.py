import os
from config import RESULTS_DIR, LOGGER




import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)


def check_tensorflow_availability():
    try:
        import tensorflow
        from tensorflow.keras.models import load_model
        import modeling
        import training
        import bayesian_tuning
        return True, load_model, modeling, training, None, bayesian_tuning
    except Exception as e:
        LOGGER.warning(f"TensorFlow indisponível. Partes de deep learning serão ignoradas. Detalhe: {e}")
        return False, None, None, None, None, None


def _extract_hparams_dict(hps_obj):
    if isinstance(hps_obj, dict):
        return hps_obj
    values_attr = getattr(hps_obj, 'values', None)
    if isinstance(values_attr, dict):
        return values_attr
    return None


def create_model_from_hyperparameters(best_hps_dict, input_shape):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.optimizers import Adam, Nadam, RMSprop, SGD
    from tensorflow.keras.metrics import AUC, Precision, Recall, PrecisionAtRecall

    model = Sequential()
    model.add(Input(shape=input_shape))

    num_layers = best_hps_dict.get('num_layers')

    for i in range(num_layers):
        units = best_hps_dict.get(f'units_layer_{i}')
        activation = best_hps_dict.get(f'activation_{i}')
        l2_reg = best_hps_dict.get(f'l2_reg_{i}')

        model.add(Dense(units, activation=activation, kernel_regularizer=l2(l2_reg)))

        if best_hps_dict.get(f'batch_norm_{i}'):
            model.add(BatchNormalization())

        dropout_rate = best_hps_dict.get(f'dropout_{i}')
        model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='sigmoid'))

    optimizer_choice = best_hps_dict.get('optimizer')
    learning_rate = best_hps_dict.get('learning_rate')

    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_choice == 'nadam':
        optimizer = Nadam(learning_rate=learning_rate)
    elif optimizer_choice == 'sgd':
        optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        optimizer = RMSprop(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            AUC(name='auc'),
            AUC(curve='PR', name='pr_auc'),
            Precision(name='precision'),
            Recall(name='recall'),
            PrecisionAtRecall(0.80, name='precision_at_recall_80')
        ]
    )

    return model


def create_model_from_best_hps(best_hps, input_shape):
    best_hps_dict = _extract_hparams_dict(best_hps)
    if best_hps_dict is None:
        try:
            best_hps_dict = dict(best_hps)
        except Exception:
            raise TypeError(f"Hiperparâmetros em formato inesperado: {type(best_hps)}")
    return create_model_from_hyperparameters(best_hps_dict, input_shape)


def retrain_final_model(best_hps, x_train_full, y_train_full, model_name="MLP_Tuned_Final"):
    from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback
    from config import DEFAULT_BATCH_SIZE, DEFAULT_FINAL_TRAINING_EPOCHS
    import time

    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("RETREINANDO MELHOR MODELO COM TODOS OS DADOS")
    LOGGER.info("=" * 80)
    LOGGER.info(f"  - Train shape: {x_train_full.shape}")

    input_shape = (x_train_full.shape[1],)

    hps_dict = _extract_hparams_dict(best_hps)
    if hps_dict is not None:
        final_model = create_model_from_hyperparameters(hps_dict, input_shape)
    else:
        final_model = create_model_from_hyperparameters(best_hps, input_shape)

    epoch_logger = LambdaCallback(
        on_epoch_end=lambda epoch, logs: LOGGER.info(
            f"Época {epoch+1}/{DEFAULT_FINAL_TRAINING_EPOCHS} - loss: {logs['loss']:.4f} - "
            f"accuracy: {logs.get('accuracy', 0):.4f} - "
            f"precision: {logs.get('precision', 0):.4f}"
        ) if (epoch + 1) % 5 == 0 else None
    )

    early_stop = EarlyStopping(
        monitor='loss',
        patience=20,
        restore_best_weights=True,
        verbose=0
    )

    if hps_dict is not None:
        batch_size = hps_dict.get('batch_size', DEFAULT_BATCH_SIZE)
    elif hasattr(best_hps, 'get'):
        try:
            batch_size = best_hps.get('batch_size')
        except KeyError:
            batch_size = DEFAULT_BATCH_SIZE
    else:
        batch_size = DEFAULT_BATCH_SIZE

    LOGGER.info(f"Usando batch_size: {batch_size}")
    LOGGER.info(f"Epochs para treinamento final: {DEFAULT_FINAL_TRAINING_EPOCHS}")
    LOGGER.info("Iniciando treinamento final...")

    retrain_start = time.time()
    history = final_model.fit(
        x_train_full, y_train_full,
        epochs=DEFAULT_FINAL_TRAINING_EPOCHS,
        batch_size=batch_size,
        callbacks=[early_stop, epoch_logger],
        verbose=0
    )

    retrain_time = time.time() - retrain_start
    LOGGER.info(f"Treinamento final concluído em {retrain_time/60:.2f} minutos")
    LOGGER.info(f"Épocas executadas: {len(history.history['loss'])}")

    final_model_path = os.path.join(RESULTS_DIR, "modelos", f"{model_name}.keras")
    final_model.save(final_model_path)
    LOGGER.info(f"Modelo final salvo em: {final_model_path}")

    return final_model


def load_best_mlp_model(load_model):
    tuned_path = os.path.join(RESULTS_DIR, "modelos", "MLP_Tuned_Final.keras")
    bayesian_path = os.path.join(RESULTS_DIR, "modelos", "MLP_Bayesian_Final.keras")

    if os.path.exists(bayesian_path):
        LOGGER.info("Carregando modelo MLP Bayesian...")
        return load_model(bayesian_path), "MLP_Bayesian"
    elif os.path.exists(tuned_path):
        LOGGER.info("Carregando modelo MLP Tuned...")
        return load_model(tuned_path), "MLP_Tuned"

    return None, None
